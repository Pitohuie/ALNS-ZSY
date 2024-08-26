@dataclass
class CustomEVRPInstance:
    # ... [All your existing attributes and methods]

    def __post_init__(self):
        # Initialize indices and other attributes as before...
        # ... your existing code ...

        # Create a distance matrix for all locations
        coords = np.array([(loc.x, loc.y) for loc in self.locations])
        self.d_ij = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)

    def distance(self, customer_i: int, customer_j: int) -> float:
        """
        Calculates the distance between two customers based on their location indices.
        """
        location_i = self.locations[customer_i]
        location_j = self.locations[customer_j]

        # Calculate Euclidean distance between the two locations
        return ((location_i.x - location_j.x) ** 2 + (location_j.y - location_j.y) ** 2) ** 0.5

    # Your existing methods...

