# src/routing_simple.py
# Simple rule-based router: input predicted_label(s), user_text, lat/lon -> department
def route(pred_labels, user_text=None, lat=None, lon=None):
    # pred_labels: list like ['pothole']
    text = (user_text or "").lower()
    # label-first rules
    if 'pothole' in pred_labels:
        return 'Roads'
    if 'garbage' in pred_labels:
        return 'Sanitation'
    if 'water_leak' in pred_labels or 'leak' in text or 'burst' in text:
        return 'Water'
    if 'broken_light' in pred_labels or 'streetlight' in text or 'lamp' in text:
        return 'Electrical'
    if 'tree_fall' in pred_labels:
        return 'Parks' if 'park' in text else 'Roads'
    return 'Other'

#usage example
from src.routing_simple import route
route(['pothole'],'near school crossing', 28.7, 77.1)  # -> 'Roads'
