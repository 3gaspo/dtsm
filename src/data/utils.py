import numpy as np

def get_temporal_features(date):
    """returns list of size context(=3) for a given date"""
    features = []
    hour, weekday, posan = date.hour, date.weekday(), date.timetuple().tm_yday

    features.append(np.cos((2*np.pi / 23) * hour)) # cos(hour)
    features.append(np.sin((2*np.pi / 23) * hour)) # sin
    features.append(np.cos((2*np.pi / 6)* weekday)) # cos(weekday)
    features.append(np.sin((2*np.pi / 6)* weekday)) # sin
    features.append(np.cos((2*np.pi / 365) * posan)) # cos(position in year)
    features.append(np.sin((2*np.pi / 365) * posan)) # sin

    return features


def train_test_split(values, datetimes, indiv_split=0.8, date_split=0.8, seed=None):
    """splits values and datetimes with a split among individuals and dates"""

    if seed is not None:
        np.random.seed(seed)

    if date_split is not None and date_split<1: #split dates
        dates = len(datetimes)
        stop_date = int(date_split * dates)
        dates1, dates2 = datetimes[:stop_date], datetimes[stop_date:] 

        if indiv_split is not None and indiv_split<1: #split individuals
            individuals = values.shape[0]
            stop_indiv = int(indiv_split * individuals)
            indices = np.random.permutation(individuals)
            indices1, indices2 = indices[:stop_indiv], indices[stop_indiv:]

            values1 = values[indices1, :, :stop_date]
            values2 = values[indices1, :, stop_date:]
            values3 = values[indices2, :, :stop_date]
            values4 = values[indices2, :, stop_date:]
            return {"train":(values1,dates1), "valid":(values2, dates2), "split_3":(values3, dates1), "test": (values4, dates2)}

        else:
            return {"train": (values[:,:,dates1], dates1), "test":(values[:,:,dates2], dates2)}

    elif indiv_split is not None and indiv_split<1: #split individuals
        individuals = values.shape[0]
        stop_indiv = int(indiv_split * individuals)
        indices = np.random.permutation(individuals)
        indices1, indices2 = indices[:stop_indiv], indices[stop_indiv:]

        values1 = values[indices1, :, :]
        values2 = values[indices2, :, :]
        return {"train":(values1, dates1), "test" :(values2, dates2)}
    
    else:
        return {"train": (values, datetimes)}

