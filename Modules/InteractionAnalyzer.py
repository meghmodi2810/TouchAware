import math

class InteractionAnalyzer:
    def __init__(self, safe_threshold=50):  # pixels
        self.safe_threshold = safe_threshold  # distance in pixels for "touch"

    def euclidean_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def analyze(self, hand_points, body_points):
        """
        hand_points: List of tuples like [(x1, y1), (x2, y2), ...]
        body_points: Dict like {'chest': (x, y), 'waist': (x, y), 'shoulder': (x, y)}
        """
        zone_touched = "none"
        min_distance = float("inf")

        for zone_name, zone_point in body_points.items():
            for hand_point in hand_points:
                dist = self.euclidean_distance(hand_point, zone_point)
                if dist < self.safe_threshold and dist < min_distance:
                    min_distance = dist
                    zone_touched = zone_name

        return zone_touched, min_distance
