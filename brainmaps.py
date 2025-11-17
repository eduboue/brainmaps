import copy
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
import nrrd
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


# ---- 1) Alias and Literals -----
ChannelMode = Literal["ch1", "ch2", "ndi_ch1", "ndi_ch2", "log_ratio_ch1", "log_ratio_ch2"]
Tail = Literal["two-sided", "greater", "less"]
DownMethod = Literal["local_mean", "gaussian_stride"]
_FILE_PAT = re.compile(r'^(?P<base>.+)_(?P<ch>01|02)(?:_.+)?\.nrrd$')

"""
Guide

ChannelMode meanings:
Channel Mde refers to what channels you want to run the stats on. 
   "ch1": Use channel 1 as-is
   "ch2": Use channel 2 as-is
   "ndi_ch1": Use normliazed differences for ch1: (ch1 - ch2) / (ch1 + ch2)  (emphasis: ch1 as the "signal")
   "ndi_ch2": Use normliazed differences for ch2: (ch2 - ch1) / (ch1 + ch2)  (emphasis: ch2 as the "signal")
   "log_ratio_ch1": Use normliazed differences for ch1: log((ch1+eps) / (ch2+eps))
   "log_ratio_ch2": Use normliazed differences for ch2: log((ch2+eps) / (ch1+eps))
   
Tail meanings:
Tail refers to the tails for the permutation test.   
    "two-sided" is a classic two sided test when you want to see which are below and above the mean
    "greater" refers to a one-sided test when your expectation is that the deviation is above the mean
    "less" refers to a one-sided twat when your expecation is that the deviation is below the mean
"""


# ---- 2) Config dataclass for constructing experiments ----
@dataclass(frozen=True)
class Config:
    """_summary_
    Dataclass that makes a config file to load multiple experiments. 
    
    Attributes:
        filepath (str): Holds filepath
        condition (str): This is the condition flag used in the stats. The Analysis suite will require at least two conditions
        zfirst (bool): The Analysis suite will require each stack to be indexed with the z-axis first to facilitate quick indexing.
            If not (z, x, y), a _swap_axis function will be applied individually or on both channels depending on input. 
            Can also be flipped latter using flip_axis()
    """
    filepath: str
    condition: str 
    zfirst: bool

# ---- 3) Experiment class for holding individual experiments ----
class Experiment:
    """
    Represents a single imaging experiment consisting of one or two aligned (registered) NRRD image channels,
    along with associated metadata such as filename, experimental condition, and axis orientation. The program 
    is written using FIJI save nrrd as input.

    Each channel is stored as a tuple of:
        - A NumPy ndarray containing the imaging data.
        - A metadata dictionary from the NRRD header.

    The class provides functionality to:
        • Load experiment data directly from one or two NRRD files via `from_file()`.
        • Optionally reorder image axes so that the Z-axis is first, either during initialization
        (by setting `zfirst=False`) or later using `flip_axis()`.
        • Maintain information about whether the Z-axis is already first (`zfirst` flag).

    Attributes:
        filename (str): The base filename of the experiment (without channel suffix).
        condition (str | None): Experimental condition label (e.g., "Control", "Treated").
        channel_one (Tuple[NDArray[Any], Dict[str, Any]]):
            First image channel and its NRRD metadata.
        channel_two (Tuple[NDArray[Any], Dict[str, Any]] | None):
            Second image channel and its NRRD metadata, if present.
        zfirst (bool): Whether the Z-axis is already the first axis in the image arrays.

    Example:
        >>> exp = Experiment.from_file("data/sample", condition="Control", zfirst=False)
        >>> exp.zfirst
        True
        >>> exp.flip_axis()  # Swap axes again if needed
    """
    def __init__(self,
                 channel_one: Tuple[NDArray[Any], Dict[str, Any]],
                 channel_two: Tuple[NDArray[Any], Dict[str, Any]] | None,
                 filename: str,
                 condition: str = None,
                 zfirst: bool = True
                 ):
        
        self.filename = filename
        self.condition = condition
        self.channel_one = channel_one
        self.channel_two = channel_two
        self.zfirst = zfirst

        # If z-axis not indexed first, flip the axis and update the zfirst parameter
        if not zfirst:
            self._swap_axes()

    @classmethod
    def from_file(cls,
                  filepath: str,
                  condition: str | None = None,
                  zfirst: bool = True
                  ) -> "Experiment":
        """
        Class method to construct the Experiment class. 

        Args:
            filepath (str): The filepath where the images are found. This assumes naming schemes are such that *_01.nrrd is the first channel
                and, if there is a second channel, *_02.nrrd, where * in both are the same
            condition (str | None, optional): The condition of the Experiment. This is used later in stats. Defaults to None.
            zfirst (bool, optional): Bool indicating whether the stack is indexed in (z, x, y) orinetation. If zfirst set to False, it will flip. Defaults to True.

        Raises:
            TypeError: If no NRRD file is found at the specified location, TypeError is rasied.

        Returns:
            Experiment: The Experiment is a class that described one brain imaging experiment. 
            The class assumes tERK and pERK staining (see Randlett et al., 2015, Nature Methods and Kozol et al., 2025 Science Advances).
            This Experiment will hold either one or two nrrd stacks as NumPy arrays, and their assocaited metadata and has methods for swapping 
                the z-axis if needed/desired for both the NumPy array and metadata. Also keeps track of z-direction and condition for each experiment
        """

        # Load files from path. Support either plain names like "<base>_01.nrrd"/"_02.nrrd"
        # or names with a shared suffix like "<base>_01_<suffix>.nrrd" and "<base>_02_<same suffix>.nrrd".
        # Users still pass `filepath` WITHOUT the channel or suffix, e.g., ".../brain_one".
        base_dir  = os.path.dirname(filepath) or "."
        base_name = os.path.basename(filepath)
        try:
            candidates = [f for f in os.listdir(base_dir) if f.startswith(base_name + "_") and f.endswith(".nrrd")]
        except FileNotFoundError:
            raise TypeError(f"Directory not found for base path: {filepath!r}")
        
        # Regex: base_<ch>(_<suffix>)?.nrrd where ch is 01 or 02. Capture suffix (may be empty).
        pat = re.compile(rf"^{re.escape(base_name)}_(?P<ch>01|02)(?P<suffix>(?:_.+?)?)\.nrrd$")
        matches: dict[str, tuple[str,str]] = {}
        for fname in candidates:
            m = pat.match(fname)
            if not m:
                continue
            ch = m.group('ch')
            suffix = m.group('suffix') or ""
            # store first match per channel (prefer exact pair by suffix later)
            full = os.path.join(base_dir, fname)
            matches.setdefault(ch, [])
            matches[ch].append((suffix, full))
        
        if not matches:
            raise TypeError("Missing required NRRD files.")
        
        # Choose a pair with matching suffix between 01 and 02 if both channels exist.
        channel_one = channel_two = None
        if "01" in matches and "02" in matches:
            # Build dict of suffix -> path for each channel
            map01 = dict(matches["01"])
            map02 = dict(matches["02"])
            # Find any shared suffix (prefer longest, to bias more specific names)
            shared = sorted(set(map01).intersection(map02), key=len, reverse=True)
            if shared:
                suf = shared[0]
                channel_one = nrrd.read(map01[suf])
                channel_two = nrrd.read(map02[suf])
        # Fallbacks: if only 01 exists, load it. If 01/02 exist but no shared suffix, try bare files.
        if channel_one is None:
            # Try bare file first
            bare01 = os.path.join(base_dir, f"{base_name}_01.nrrd")
            if os.path.exists(bare01):
                channel_one = nrrd.read(bare01)
            elif "01" in matches and matches["01"]:
                # pick the first available 01
                channel_one = nrrd.read(matches['01'][0][1])
        if channel_two is None and "02" in matches:
            bare02 = os.path.join(base_dir, f"{base_name}_02.nrrd")
            if os.path.exists(bare02):
                channel_two = nrrd.read(bare02)
            elif matches["02"]:
                channel_two = nrrd.read(matches['02'][0][1])
        if channel_one is None and channel_two is None:
            raise TypeError("Missing required NRRD files.")

        # Set filename using path.basename
        filename = os.path.basename(filepath)
        return cls(channel_one, channel_two, filename, condition=condition, zfirst=zfirst)

    def _swap_axes(self) -> None:
        """
        Public method to swap Z axis to first position in both channels.
        Calls on _swap_single instance method and applies to one or both channels
        """
        self.channel_one = self._swap_single(self.channel_one)
        if self.channel_two is not None:
            self.channel_two = self._swap_single(self.channel_two)
        self.zfirst = True

    @staticmethod
    def _swap_single(channel: Tuple[NDArray[Any], Dict[str, Any]]) -> Tuple[NDArray[Any], Dict[str, Any]]:
        """ 
        Internal static method to swap a single channel. _swap_axes() calls on this for both axes .
        
        Args:
            Takes in a channel (Tuple with NDArray as stack and dict as metadata)
        Raises:
            None
        Returns:
            Tuple array like input except NDArray z-axis flipped and assocaited metadata as well
        """
        arr, hdr = channel
        arr = np.moveaxis(arr, -1, 0)
        hdr["sizes"] = np.roll(hdr["sizes"], shift=1)
        hdr["space directions"] = np.fliplr(np.flip(hdr["space directions"], axis=0))
        if hasattr(hdr, "labels"):
            hdr["labels"] = np.roll(hdr["labels"], shift=1)
        return arr, hdr

    def flip_axis(self) -> None:
        """Public method to flip axes after initialization."""
        self._swap_axes()
        
# ---- 4) Helper to build experiments from configs ----
def make_experiment(cfg: Config) -> Experiment:
    """ Quick helper method to load config file and apply to the AnalaysisSuite.from_configs() classmethod """ 
    return (Experiment.from_file(filepath=cfg.filepath, 
                                 condition=cfg.condition,
                                 zfirst=cfg.zfirst))

def build_configs_from_directory(
    root_dir: str,
    zfirst: bool = True,
    write_json: str | None = None,
) -> list[Config]:
    """
    Scan a directory structured as:
        root_dir/
            <condition-1>/
                <base>_01[_<suffix>].nrrd
                <base>_02[_<suffix>].nrrd
                ...
            <condition-2>/
                ...

    Returns:
        List[Config] where:
            - filepath is the FULL BASE PATH (no _01/_02 or suffix), e.g. ".../Surface_Fed/brain_one"
            - condition is the condition folder name
            - zfirst is the provided default for all

    If write_json is provided (a path), writes a JSON file with an array of
    { "filepath": "...", "condition": "...", "zfirst": true/false }.

    Notes:
        - Only immediate subfolders of `root_dir` are treated as conditions.
        - De-duplicates multiple files that correspond to the same base.
        - Works with both plain and decorated NRRD names.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir!r}")

    configs: list[Config] = []

    # Each immediate subdirectory is treated as a condition
    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        condition = entry.name
        cond_dir = entry.path

        # Collect unique bases discovered in this condition directory
        bases_seen: set[str] = set()

        for f in os.scandir(cond_dir):
            if not f.is_file():
                continue
            if not f.name.lower().endswith(".nrrd"):
                continue

            m = _FILE_PAT.match(f.name)
            if not m:
                continue

            base = m.group("base")

            # Build absolute base path (without _01/_02 and without any suffix)
            base_full = os.path.abspath(os.path.join(cond_dir, base))

            if base_full in bases_seen:
                continue
            bases_seen.add(base_full)

            # Create Config entry
            configs.append(Config(filepath=base_full, condition=condition, zfirst=zfirst))

    # Optional: write JSON to disk
    if write_json is not None:
        payload = [
            {"filepath": c.filepath, "condition": c.condition, "zfirst": bool(c.zfirst)}
            for c in configs
        ]
        with open(write_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote {len(configs)} configs to {write_json}")

    return configs

# ---- 5) Orchestrator ----
@dataclass
class AnalysisSuite:
    """
    Container class for managing and analyzing multiple `Experiment` objects.

    The `AnalysisSuite` stores a collection of experiments, provides utilities to validate 
    their compatibility for analysis, compute average images for each channel, and apply 
    batch operations such as axis flipping. It ensures that all experiments meet the required 
    conditions (e.g., same dimensions, correct axis order, and exactly two experimental conditions) 
    before an analysis can be run.

    Attributes:
        experiments (List[Experiment]):
            A list of `Experiment` instances to include in the analysis.
        _OK_to_run (bool):
            Internal flag indicating whether all validation checks have passed.
        _avg_one (Optional[NDArray[Any]]):
            Cached average image for channel one, computed on first access.
        _avg_two (Optional[NDArray[Any]]):
            Cached average image for channel two, computed on first access, 
            or `None` if no experiments contain channel two.

    Properties:
        avg_channel_one (NDArray[Any]):
            The average image for channel one, computed across all experiments.
        avg_channel_two (Optional[NDArray[Any]]):
            The average image for channel two, or `None` if unavailable.

    Methods:
        flip_axes():
            Applies axis flipping to all experiments in the suite.
        check_analysis_conditions():
            Verifies that all experiments have:
                - Exactly two distinct experimental conditions.
                - The Z-axis as the first dimension.
                - Matching dimensions for all images in channel one.
            Sets `_OK_to_run` to True if all checks pass.
        from_configs(configs: Iterable[Config]) -> AnalysisSuite:
            Class method to build an `AnalysisSuite` from a list of `Config` objects.

    Example:
        >>> configs = [
        ...     Config(filepath="data/sample1", condition="Control"),
        ...     Config(filepath="data/sample2", condition="Treatment")
        ... ]
        >>> suite = AnalysisSuite.from_configs(configs)
        >>> suite.check_analysis_conditions()
        All parameters check out. Proceed to setup analysis run
        >>> avg_img = suite.avg_channel_one
    """
    
    experiments: List["Experiment"]
    _OK_to_run: bool = False

    def get_conditons(self) -> set:
        return {experiment.condition for experiment in self.experiments}
    
    def flip_axes(self) -> None:
        """ Flips axes of all Experiments in AnalysisSuite container"""
        for experiment in self.experiments:
            experiment.flip_axis()

    def check_anlysis_conditions(self):
        """
        Method that checks to ensure that an Analysis can be run. 
        Checks that there are 2 conditions, the z-axis is listed first for all, and the dimensions are the same for all files.

        Raises:
            TypeError: TypeError raised if there are more than 2 conditions
            TypeError: TypeError raised if any Experiment file is not indexed with z-axis first
            TypeError: TypeError raised if the Experiments are not in the same 3D space (dimensions not same among Experiments)
        """
        condition_check = {experiment.condition for experiment in self.experiments}
        if len(condition_check) != 2:
            raise TypeError("You must have two different conditions to run an anlysis")
        zfirst_check = {experiment.zfirst for experiment in self.experiments}
        if False in zfirst_check:
            raise TypeError("All experiments must be in a zfirst order")
        dimensions_check = [hdr["sizes"] for _, hdr in (exp.channel_one for exp in self.experiments)]
        all_equal = all(np.array_equal(dimensions_check[0], arr) for arr in dimensions_check[1:])
        if not all_equal:
            raise TypeError("Dimensions among arrays do not match. \nPlease check your dimensions and ensure they are all in the same space. \nCheck whether your images have been registered, and whether they are all in the z-first orientation")
        self._OK_to_run = True
        print("All parameters check out. Proceed to setup analysis run")
                           
                              
    @classmethod
    def from_configs(cls, configs: Iterable[Config]) -> "AnalysisSuite":
        """ 
        Class method to take config file containing filepath, condition, and zfirst status 
        and load to Analysis Suite class to make individual Experiment classes
        """
        return cls([make_experiment(c) for c in configs])
    
# ------6) PermutationResults dataclass ------
@dataclass
class PermutationResult:
    effect_map: NDArray[Any]          # (Z, X, Y) observed difference in means: mean(g2) - mean(g1)
    p_map: NDArray[Any]               # (Z, X, Y) permutation p-values
    q_map: NDArray[Any]               # (Z, X, Y) BH-FDR q-values
    sig_mask: NDArray[Any]            # (Z, X, Y) boolean, q <= alpha
    group_means: Tuple[NDArray, NDArray]  # (mean_g1, mean_g2), each (Z, X, Y)
    group_ns: Tuple[int, int]       # (n1, n2)
    params: Dict[str, Any]          # bookkeeping (tail, n_perm, alpha, rng_seed)
    
    def __str__(self) -> str:
        # Shape info
        shape = self.effect_map.shape
        n_sig = int(self.sig_mask.sum())
        total = self.sig_mask.size

        # Summaries
        p_min, p_med, p_max = (
            float(np.min(self.p_map)),
            float(np.median(self.p_map)),
            float(np.max(self.p_map)),
        )
        q_min, q_med, q_max = (
            float(np.min(self.q_map)),
            float(np.median(self.q_map)),
            float(np.max(self.q_map)),
        )

        return (
            "PermutationResult\n"
            f"  Groups: n1={self.group_ns[0]}, n2={self.group_ns[1]}\n"
            f"  Effect map shape: {shape}\n"
            f"  Params: tail={self.params.get('tail')}, "
            f"n_perm={self.params.get('n_perm')}, "
            f"alpha={self.params.get('alpha')}, "
            f"seed={self.params.get('rng_seed')}\n"
            f"  P-values: min={p_min:.4g}, median={p_med:.4g}, max={p_max:.4g}\n"
            f"  Q-values: min={q_min:.4g}, median={q_med:.4g}, max={q_max:.4g}\n"
            f"  Significant voxels: {n_sig}/{total} "
            f"({100*n_sig/total:.2f}%) at q <= {self.params.get('alpha')}\n"
        )
    
# ----- 7) Helper functions --------
def _get_two_groups_subset(suite: "AnalysisSuite", 
                           groups: Tuple[str, str]
                           ) -> List["Experiment"]:
    
    g1, g2 = groups
    selected = [e for e in suite.experiments if e.condition in {g1, g2}]
    if len({e.condition for e in selected}) != 2:
        raise ValueError(
            f"After subsetting, need exactly two groups; got {sorted({e.condition for e in selected})}"
        )
    return selected

def _select_or_compute_image(exp: "Experiment",
                             mode: ChannelMode,
                             eps: float = 1e-6
                             ) -> np.ndarray:
    # pull arrays
    ch1 = exp.channel_one[0] if exp.channel_one is not None else None
    ch2 = exp.channel_two[0] if exp.channel_two is not None else None

    if mode in ("ch1", "ndi_ch1", "log_ratio_ch1"):
        if ch1 is None:
            raise ValueError(f"Experiment {exp.filename} missing channel 1.")
    if mode in ("ch2", "ndi_ch2", "log_ratio_ch2"):
        if ch2 is None:
            raise ValueError(f"Experiment {exp.filename} missing channel 2.")

    if mode == "ch1":
        return ch1
    if mode == "ch2":
        return ch2

    # NDI variants
    if mode == "ndi_ch1":
        denom = ch1 + ch2
        return (ch1 - ch2) / (denom + eps)
    if mode == "ndi_ch2":
        denom = ch1 + ch2
        return (ch2 - ch1) / (denom + eps)

    # log-ratio variants
    if mode == "log_ratio_ch1":
        return np.log((ch1 + eps) / (ch2 + eps))
    if mode == "log_ratio_ch2":
        return np.log((ch2 + eps) / (ch1 + eps))

    raise ValueError(f"Unknown channel mode: {mode}")

def _affine_fit_to_template(B: NDArray[Any], 
                            A: NDArray[Any]
                            ) -> NDArray[np.float64]:
    """
    Solve min_{a,b} || a*B + b - A ||_2^2 in closed form:
        a = cov(B,A) / var(B)
        b = mean(A) - a * mean(B)
    """
    x = B.reshape(-1).astype(np.float64)
    y = A.reshape(-1).astype(np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    x_var = np.var(x)
    if x_var < 1e-12:
        # degenerate (flat) image; just shift to the template mean
        return np.full_like(B, y_mean, dtype=np.float64)
    cov = np.mean((x - x_mean) * (y - y_mean))
    a = cov / (x_var + 1e-20)
    b = y_mean - a * x_mean
    out = a * B + b
    return out

def _bh_fdr(p: NDArray[Any], 
            alpha: float = 0.05
            ) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Benjamini–Hochberg FDR control.
    Returns:
        q: q-values (same shape as p)
        mask: boolean for q <= alpha
    """
    p_flat = p.reshape(-1)
    n = p_flat.size
    order = np.argsort(p_flat, kind="mergesort")  # stable
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)

    # Compute BH adjusted p-values (q-values)
    q_flat = p_flat * n / ranks
    # Ensure monotonicity of q-values (non-decreasing after ordering)
    q_sorted = q_flat[order]
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_flat[order] = q_sorted
    q = q_flat.reshape(p.shape)

    mask = q <= alpha
    return q, mask

def _perm_pvalues_streaming(X1: NDArray[Any],  # shape (n1, Z, X, Y)
                            X2: NDArray[Any],  # shape (n2, Z, X, Y)
                            n_perm: int,
                            tail: Tail,
                            rng: np.random.Generator,
                            ) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """
    Streaming permutation p-values for the difference in means (mean(g2) - mean(g1)).
    Avoids storing the full null distribution.

    Returns:
        effect_map: observed effect (Z, X, Y)
        p_map: permutation p-values (Z, X, Y)
        var_map: pooled variance estimate (optional for diagnostics)
    """
    n1, n2 = X1.shape[0], X2.shape[0]
    Z, X, Y = X1.shape[1:]
    V = Z * X * Y

    # Flatten spatial dims for speed: (n, V)
    A = X1.reshape(n1, V).astype(np.float16, copy=False)
    B = X2.reshape(n2, V).astype(np.float16, copy=False)

    # Observed stats
    mean1 = A.mean(axis=0)
    mean2 = B.mean(axis=0)
    effect = (mean2 - mean1)  # observed difference
    # simple pooled variance (not strictly needed for permutation test, but useful)
    var1 = A.var(axis=0, ddof=1) if n1 > 1 else np.zeros_like(mean1)
    var2 = B.var(axis=0, ddof=1) if n2 > 1 else np.zeros_like(mean2)
    var_pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1)

    # Combine for permutation
    C = np.vstack([A, B])  # (n1+n2, V)
    n = C.shape[0]
    idx = np.arange(n, dtype=int)

    # Streaming exceedance counts
    ge_count = np.zeros(V, dtype=np.uint32)  # count of >= observed (for right tail)
    le_count = np.zeros(V, dtype=np.uint32)  # count of <= observed (for left tail)

    # Precompute constants for fast diff-of-means updates
    inv_n1 = 1.0 / n1
    inv_n2 = 1.0 / n2

    for _ in tqdm(range(n_perm)):
        rng.shuffle(idx)
        g2_idx = idx[:n2]   # assign first n2 to group 2
        g1_idx = idx[n2:]   # remaining to group 1

        # Means under permutation
        perm_mean2 = C[g2_idx].mean(axis=0)
        perm_mean1 = C[g1_idx].mean(axis=0)
        perm_effect = perm_mean2 - perm_mean1

        # Update counts
        # Right tail: perm_effect >= effect
        ge_count += (perm_effect >= effect)
        # Left tail: perm_effect <= effect
        le_count += (perm_effect <= effect)

    # P-values with +1 smoothing
    if tail == "greater":
        p = (ge_count + 1.0) / (n_perm + 1.0)
    elif tail == "less":
        p = (le_count + 1.0) / (n_perm + 1.0)
    else:  # two-sided
        # symmetric two-sided using max of tail counts
        tail_counts = np.minimum(ge_count, le_count)
        # Two-sided p ≈ 2 * min-tail, but bounded by 1. Use permutation smoothing too.
        p = 2.0 * (tail_counts + 1.0) / (n_perm + 1.0)
        p = np.minimum(p, 1.0)

    # Reshape back to (Z, X, Y)
    effect_map = effect.reshape(Z, X, Y)
    p_map = p.reshape(Z, X, Y)
    var_map = var_pooled.reshape(Z, X, Y)
    return effect_map, p_map, var_map

# ------ 8) Public pipeline functions -------
def downsample_volume(vol: NDArray[Any],              # shape (Z, X, Y) or (Z, H, W)
                      factors: Tuple[int, int, int] = (1, 2, 2),
                      method: DownMethod = "local_mean",
                      anti_alias_sigma: Tuple[float, float, float] | None = None,
                      dtype: np.dtype = np.float32,
                      ) -> np.ndarray:
    """
    Downsample a 3D volume with either local-mean blocks or Gaussian+stride.
    - local_mean preserves mean intensity exactly within blocks.
    - gaussian_stride applies an anti-aliasing blur then strides by the factor.

    factors: integer downsample factors per axis (Z, X, Y).
    anti_alias_sigma: if None, defaults to (f/2) for gaussian_stride.
    """
    assert vol.ndim == 3, "Expected (Z, X, Y)"
    zf, xf, yf = factors
    vol = vol.astype(dtype, copy=False)

    if method == "local_mean":
        # Pad so dimensions are multiples of factors (edge replicate)
        Z, X, Y = vol.shape
        Zp = (zf - (Z % zf)) % zf
        Xp = (xf - (X % xf)) % xf
        Yp = (yf - (Y % yf)) % yf
        if Zp or Xp or Yp:
            vol = np.pad(vol,
                         ((0, Zp), (0, Xp), (0, Yp)),
                         mode="edge")
        # Reshape and average
        Z2, X2, Y2 = vol.shape
        vol = vol.reshape(Z2 // zf, zf, X2 // xf, xf, Y2 // yf, yf).mean(axis=(1, 3, 5))
        return vol.astype(dtype, copy=False)

    elif method == "gaussian_stride":
        if anti_alias_sigma is None:
            anti_alias_sigma = (max(zf, 1)/2.0, max(xf, 1)/2.0, max(yf, 1)/2.0)
        blurred = gaussian_filter(vol, sigma=anti_alias_sigma, mode="nearest")
        return blurred[::zf, ::xf, ::yf].astype(dtype, copy=False)

    else:
        raise ValueError(f"Unknown method: {method}")

def prepare_for_permutation(suite: "AnalysisSuite",
                            groups: Tuple[str, str],
                            channel_mode: ChannelMode,
                            do_brightness_affine: bool = True,
                            eps: float = 1e-6,
                            *,
                            downsample: bool = True,
                            ds_factors: Tuple[int, int, int] = (1, 2, 2),
                            ds_method: DownMethod = "local_mean",
                            ) -> "AnalysisSuite":
    """
    Validate + subset an AnalysisSuite to two groups, create a single processed
    image per experiment according to `channel_mode`, optionally brightness-normalize
    each experiment to the template average (global affine), and return a DEEP COPY
    of the AnalysisSuite whose experiments now contain:
        - channel_one: the finalized image (np.ndarray) and a minimal header
        - channel_two: None
        - same condition, zfirst=True

    The returned suite is ready to feed into your (static) permutation/FDR function.
    """
    # 1) Ensure the suite is validated
    ok_flag = getattr(suite, "_OK_to_run", False)
    if not ok_flag:
        # If 
        # try to call it if present, otherwise raise.
        if hasattr(suite, "check_anlysis_conditions"):
            suite.check_anlysis_conditions()
        else:
            raise RuntimeError("AnalysisSuite not validated and no check method available.")
    
    # 2) Subset to the two requested groups
    selected = _get_two_groups_subset(suite, groups)

    # 3) Make the processed stack per experiment (list of arrays)
    proc_images: List[np.ndarray] = []
    for exp in selected:
        img = _select_or_compute_image(exp, channel_mode, eps=eps)
        proc_images.append(img)
    
    # 4) Downsample images
    if downsample:
        new_proc_images: List[np.ndarray] = []
        for stack in proc_images:
            # for gaussian_stride you can set anti_alias_sigma if you want:
            # anti_alias_sigma=(0.5, 1.0, 1.0)
            img = downsample_volume(stack, factors=ds_factors, method=ds_method)

            new_proc_images.append(img)
        proc_images = new_proc_images
        
    for e in proc_images:
        print(e.shape)

    # 5) Optional global affine brightness normalization
    if do_brightness_affine:
        template = np.mean(np.stack(proc_images, axis=0), axis=0)
        proc_images = [_affine_fit_to_template(B, template) for B in proc_images]

    # 6) Build a deep-copied AnalysisSuite with processed images as channel_one, channel_two=None
    new_suite = copy.deepcopy(suite)
    # Keep only selected experiments, in same order as `selected`
    new_suite.experiments = []

    for exp, img in zip(selected, proc_images):
        # Construct a minimal NRRD-like header carrying sizes, so other utilities won't break
        hdr = {"sizes": np.array(img.shape, dtype=int)}
        new_exp = Experiment(
            channel_one=(img, hdr),
            channel_two=None,
            filename=exp.filename,
            condition=exp.condition,
            zfirst=True
        )
        new_suite.experiments.append(new_exp)

    # The new suite is “ready” for stats (it passed validation earlier, and we’re zfirst)
    new_suite._OK_to_run = True
    # Optionally clear cached averages if your class uses them
    if hasattr(new_suite, "_avg_one"):
        new_suite._avg_one = None
    if hasattr(new_suite, "_avg_two"):
        new_suite._avg_two = None

    return new_suite

def stack_by_group(suite: "AnalysisSuite",
                    groups: Tuple[str, str]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience helper to produce two 4D stacks (n, z, x, y) of the finalized
    images for the two groups. Assumes each experiment has a single processed
    image in channel_one and z-first orientation.

    Returns:
        X_g1: array of shape (n1, Z, X, Y)
        X_g2: array of shape (n2, Z, X, Y)
    """
    g1, g2 = groups
    g1_imgs = []
    g2_imgs = []
    for e in suite.experiments:
        arr = e.channel_one[0]
        if e.condition == g1:
            g1_imgs.append(arr)
        elif e.condition == g2:
            g2_imgs.append(arr)
    if not g1_imgs or not g2_imgs:
        raise ValueError("Missing one or both groups when stacking by group.")
    X_g1 = np.stack(g1_imgs, axis=0)
    X_g2 = np.stack(g2_imgs, axis=0)
    return X_g1, X_g2

def run_permutation(prepared_suite: "AnalysisSuite",
                    groups: Tuple[str, str],
                    n_perm: int = 5000,
                    tail: Tail = "two-sided",
                    alpha: float = 0.05,
                    random_state: Optional[int] = None,
                    ) -> PermutationResult:
    """
    Pixel-wise permutation test (label shuffling) between two groups in a prepared AnalysisSuite.
    Assumes each experiment in `prepared_suite` holds ONE finalized image in `channel_one`
    (e.g., via `prepare_for_permutation(... )`).

    Workflow:
      1) Build stacks X_g1, X_g2 (n1, Z, X, Y), (n2, Z, X, Y).
      2) Compute observed effect = mean(g2) - mean(g1).
      3) Run streaming permutations (no huge null array).
      4) Compute p-map and BH-FDR q-map, and significance mask at alpha.

    Returns:
      PermutationResult dataclass with maps and metadata.
    """
    rng = np.random.default_rng(random_state)

    # Gather stacks
    X_g1, X_g2 = stack_by_group(prepared_suite, groups)  # shapes (n1, Z, X, Y), (n2, Z, X, Y)
    if X_g1.shape[1:] != X_g2.shape[1:]:
        raise ValueError("Group images have mismatched shapes.")
    if X_g1.ndim != 4 or X_g2.ndim != 4:
        raise ValueError("Expected stacks of shape (n, Z, X, Y).")

    # Permutation p-values
    effect_map, p_map, _var_map = _perm_pvalues_streaming(
        X_g1, X_g2, n_perm=n_perm, tail=tail, rng=rng
    )

    # FDR (BH)
    q_map, sig_mask = _bh_fdr(p_map, alpha=alpha)

    # Group means (for reporting)
    mean_g1 = X_g1.mean(axis=0)
    mean_g2 = X_g2.mean(axis=0)

    return PermutationResult(
        effect_map=effect_map,
        p_map=p_map,
        q_map=q_map,
        sig_mask=sig_mask,
        group_means=(mean_g1, mean_g2),
        group_ns=(X_g1.shape[0], X_g2.shape[0]),
        params={"tail": tail, "n_perm": n_perm, "alpha": alpha, "rng_seed": random_state},
    )
     
if __name__ == "__main__":
    # Build configs:
    # configs = build_configs_from_directory("/data/feeding", zfirst=True)
    configs = [
        Config(filepath="/Users/erikduboue/Downloads/RxFish_number_three", condition="Pachon Fed", zfirst=False),
        Config(filepath="/Users/erikduboue/Downloads/RxFish_number_three", condition="Pachon Fed", zfirst=False),
        Config(filepath="/Users/erikduboue/Downloads/RxFish_number_three", condition="Surface Fed", zfirst=False),
        Config(filepath="/Users/erikduboue/Downloads/RxFish_number_three", condition="Surface Fed", zfirst=False),
    ]
    
    suite: AnalysisSuite = AnalysisSuite.from_configs(configs)
    suite.check_anlysis_conditions()
    groups = ("Surface Fed", "Pachon Fed")
    # channel_mode options: "ch1", "ch2", "ndi_ch1", "ndi_ch2", "log_ratio_ch1", "log_ratio_ch2"
    prepared = prepare_for_permutation(suite,
                                       groups=groups,
                                       channel_mode="ndi_ch2",
                                       do_brightness_affine=True,
                                       downsample = True,
                                       ds_factors = (2,2,2),
                                       ds_method = "local_mean"
                                       )
    res = run_permutation(prepared,groups = groups, n_perm = 2000, tail = "two-sided", alpha = 0.05)
    print(res)