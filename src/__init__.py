"""
MetricMatch: select annotation-efficient subsets to estimate LLM judge reliability.

Core public API:
    from src.utils.selection_strategies import metric_matched_selection
    from src.utils.match_metrics import compute_icc_pingouin, compute_krippendorff_alpha
    from src.utils.data_loading import load_judge_scores
"""
