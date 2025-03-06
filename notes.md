# NxMap creation works!!!

But the CRS doesnt match the traces or the one used in the matcher

error:

    Creating graph
    Creating NxMap
    Loaded data in 2.9 seconds!
    [1]: Processing!
    creating nxmap from geofence
    matching trace to map
    Failed to download map for taxi 1: crs of origin EPSG:3857 must match crs of map EPSG:4326
    [2]: Processing!
    creating nxmap from geofence
    matching trace to map
    Failed to download map for taxi 2: crs of origin EPSG:3857 must match crs of map EPSG:4326
    {1: [], 2: []}

## TODO:

-   [ ] Check what crs the shp file is
-   [ ] See if we can change it to the same csr as the T-Drive data.
