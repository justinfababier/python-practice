import numpy as np
import matplotlib.pyplot as plt
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
        p0, p1 = xy_nodes[i], xy_nodes[i+1]
        d = unit(p1 - p0)
        n = np.array([-d[1], d[0]])
        left_boundary.append(p0 + (width/2)*n)
        right_boundary.append(p0 - (width/2)*n)
    
    # Stopbar node
    d = unit(xy_nodes[-1] - xy_nodes[-2])
    n = np.array([-d[1], d[0]])
    left_boundary.append(xy_nodes[-1] + (width/2)*n)
    right_boundary.append(xy_nodes[-1] - (width/2)*n)

    polygon = np.vstack([left_boundary, right_boundary[::-1]])

    # Convert back to lon/lat for plotting
    nodes_ll   = np.array([inv.transform(x,y) for x,y in xy_nodes])
    left_ll    = np.array([inv.transform(x,y) for x,y in left_boundary])
    right_ll   = np.array([inv.transform(x,y) for x,y in right_boundary])
    polygon_ll = np.array([inv.transform(x,y) for x,y in polygon])

    return nodes_ll, left_ll, right_ll, polygon_ll

def main():
    """
    Given a set of nodes that describe a lane, create the lane's geometry.
    """
    # Array of nodes in (longitude, latitude).
    # This set of nodes describes a westbound lane approaching 
    # the University & Iowa intersection in Riverside, California.
    nodes = [(-117.3396957, 33.9757438),    # Stopbar node
             (-117.3392196, 33.9757416),
             (-117.3389018, 33.9757471),
             (-117.3386457, 33.9757505)]
    
    width = 3.658  # meters (approximate 12 feet)

    nodes_arr, left_b, right_b, polygon = lane_polygon(nodes, width)

    plt.figure(figsize=(8,6))
    plt.plot(nodes_arr[:,0], nodes_arr[:,1], 'k--o', label="Centerline (end->start)")
    plt.plot(left_b[:,0], left_b[:,1], 'b-o', label="Left boundary")
    plt.plot(right_b[:,0], right_b[:,1], 'r-o', label="Right boundary")
    plt.fill(polygon[:,0], polygon[:,1], alpha=0.2, color="gray", label="Lane polygon")

    plt.gca().set_aspect('equal', 'box')
    plt.legend()
    plt.title("Lane Geometry")
    plt.show()

if __name__ == "__main__":
    main()
