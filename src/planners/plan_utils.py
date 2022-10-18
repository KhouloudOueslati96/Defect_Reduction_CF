""" Utility functions for planners """

def apply_threshold(X_row, complexity_thresh):
    new_row = X_row.to_list()
    metrics = complexity_thresh.keys()
    for i, metric in enumerate(metrics):
        threshold = complexity_thresh[metric]
        # Sometimes the thresholds are list
        if type(threshold) == list:
            threshold = threshold[-1]
        if threshold is not None:
            try:
                if new_row[i] > threshold:
                    new_row[i] = (0, threshold)
            except:
                pass
    return new_row