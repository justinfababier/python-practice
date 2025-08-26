from pyproj import Geod

# Westbound lane located at: University & Iowa, Riverside, California
nodes = [
    (-117.3396957, 33.9757438), # Stopbar node
    (-117.3392196, 33.9757416),
    (-117.3389018, 33.9757471),
    (-117.3386457, 33.9757505),
]

geod = Geod(ellps='WGS84')

def calculate_geodesic_length(nodes):
    """
    Calculate total geodesic distance along a path of nodes
    """
    total_length = 0
    for i in range(len(nodes) - 1):
        lon1, lat1 = nodes[i]
        lon2, lat2 = nodes[i + 1]
        _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
        total_length += distance
    return total_length

def main():
    length = calculate_geodesic_length(nodes)
    print(f"Road lane length: {length:.2f} meters")
    print(f"Road lane length: {length / 1000:.3f} kilometers")
    print(f"Stopbar end node: {nodes[0]}")
    print(f"Ingress start node: {nodes[-1]}")

if __name__ == "__main__":
    main()