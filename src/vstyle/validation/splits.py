"""4.6.1 - chronological partition expressed as calendar-week indices."""
from ..data.calendar import to_t_idx


def origin_sets(partition, origin_week, horizon, t_max):
    def idx(key):
        return to_t_idx(partition[key], origin_week)

    dev_end = idx("development_end")
    test_end = idx("test_end") if partition.get("test_end") else t_max - horizon
    return {
        "calibration_end": idx("calibration_end"),
        "development_end": dev_end,
        # validation labels (t+1..t+H) must be observed inside development: t + H <= dev_end
        "validation": list(range(idx("validation_start"), dev_end - horizon + 1)),
        "test": list(range(idx("test_start"), min(test_end, t_max - horizon) + 1)),
    }


def catalogue_ids(products, cutoff):
    """4.1.4 - products allowed to shape the centroids."""
    return products.loc[products["release_first"] <= cutoff, "external_code"].tolist()
