import numpy as np
import matplotlib.pyplot as plt
from pyproj import Geod
from pyproj import Transformer

def unit(v):
    return v / np.linalg.norm(v)

def lane_polygon(nodes, width, crs_in="EPSG:4326", crs_out="EPSG:32611"):
    """
    nodes: [(lon, lat), ...] polyline from end -> start
    width: lane width in meters
    crs_in: input CRS (default WGS84)
    crs_out: projected CRS (default UTM zone 11N for California)
    """
    # Transformer: geographic -> projected (meters)
    fwd = Transformer.from_crs(crs_in, crs_out, always_xy=True)
    inv = Transformer.from_crs(crs_out, crs_in, always_xy=True)
    
    # Convert lon/lat to projected coords (x,y in meters)
    xy_nodes = np.array([fwd.transform(lon, lat) for lon, lat in nodes[::-1]])

    left_boundary, right_boundary = [], []

    for i in range(len(xy_nodes)-1):
        p0, p1 = xy_nodes[i], xy_nodes[i + 1]
        d = unit(p1 - p0)
        n = np.array([-d[1], d[0]])
        left_boundary.append(p0 + (width / 2) * n)
        right_boundary.append(p0 - (width / 2) * n)
    
    # Stopbar node
    d = unit(xy_nodes[-1] - xy_nodes[-2])
    n = np.array([-d[1], d[0]])
    left_boundary.append(xy_nodes[-1] + (width / 2) * n)
    right_boundary.append(xy_nodes[-1] - (width / 2 ) * n)

    polygon = np.vstack([left_boundary, right_boundary[::-1]])

    # Convert back to lon/lat for plotting
    nodes_ll = np.array([inv.transform(x,y) for x,y in xy_nodes])
    left_ll = np.array([inv.transform(x,y) for x,y in left_boundary])
    right_ll = np.array([inv.transform(x,y) for x,y in right_boundary])
    polygon_ll = np.array([inv.transform(x,y) for x,y in polygon])

    return nodes_ll, left_ll, right_ll, polygon_ll

def point_in_polygon(point, polygon):
  """
  Check if point exists within polygon.
  """
  x, y = point
  inside = False
  n = len(polygon)
  for i in range(n):
    x0, y0 = polygon[i]
    x1, y1 = polygon[(i + 1) % n]
    if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0):
        inside = not inside
  return inside

def main():
    """
    Given a set of nodes that describe a lane, create the lane's geometry.
    """
    # Constants
    geod = Geod(ellps='WGS84')
    lane_width = 3.658  # meters (approximate 12 feet)

    # Array of nodes in (longitude, latitude) for laneID 1002
    # These set of nodes describe a westbound lane approaching the University & Iowa intersection in Riverside, California.
    laneID_1002_nodes = np.array([(-117.3396957, 33.9757438),    # Stopbar node
                                  (-117.3392196, 33.9757416),
                                  (-117.3389018, 33.9757471),
                                  (-117.3386457, 33.9757505)])

    # Heading of laneID 1002 lane
    laneID_1002_heading, _, _ = geod.inv(laneID_1002_nodes[1, 0], laneID_1002_nodes[1, 1], 
                                         laneID_1002_nodes[0, 0], laneID_1002_nodes[0, 1])
    print(f"Heading of laneID 1002: {laneID_1002_heading % 360:.2f}")  # Expected value: ~270.0 deg

    gnss_node_prev = (-117.3390457, 33.9757438)  # GNSS point - previous
    gnss_node = (-117.3391457, 33.9757438)  # GNSS point - most current
    gnss_fwd, _, _ = geod.inv(gnss_node_prev[0], gnss_node_prev[1],
                              gnss_node[0], gnss_node[1])  # GNSS heading
    print(f"GNSS heading: {gnss_fwd % 360:.2f}")  # Expected value: ~270.0 deg

    # Determine if laneID 1002 is a reasonable candidate based off heading ONLY
    # Let's assume that laneID 1002 has been already recognized as an ingress lane
    print(f"Is laneID 1002 a reasonable candidate? {np.isclose(laneID_1002_heading, gnss_fwd, 1e-2)}")
    
    # Construct polygon for lane
    laneID_1002_nodes, left_b, right_b, polygon = lane_polygon(laneID_1002_nodes, lane_width)
    inside = point_in_polygon(gnss_node, polygon)   # Determine if GNSS point is within lane boundary
    print(f"GNSS point inside lane? {inside}")

    plt.figure(figsize=(6,4))
    plt.plot(laneID_1002_nodes[:,0], laneID_1002_nodes[:,1], 'k--o', label="Centerline (end->start)")
    plt.plot(left_b[:,0], left_b[:,1], 'b-o', label="Left boundary")
    plt.plot(right_b[:,0], right_b[:,1], 'r-o', label="Right boundary")
    plt.fill(polygon[:,0], polygon[:,1], alpha=0.2, color="gray", label="Lane polygon")
    plt.plot(gnss_node[0], gnss_node[1], 'go', label="GNSS node")
    plt.plot(gnss_node_prev[0], gnss_node_prev[1], 'go', label="GNSS node (previous)")
    # plt.legend()

    plt.gca().set_aspect('equal', 'box')
    plt.title("Lane Geometry")
    plt.show()

if __name__ == "__main__":
    main()