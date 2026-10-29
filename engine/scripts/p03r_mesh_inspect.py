import traceback

out = r"E:\p03r_grid_sensitivity_20261012\mesh_inspection.txt"
f = open(out, "w")
f.write("script started\n")
f.flush()
try:
    f.write("getting ExtAPI\n"); f.flush()
    x = ExtAPI
    f.write("getting DataModel\n"); f.flush()
    x = x.DataModel
    f.write("getting Project\n"); f.flush()
    x = x.Project
    f.write("getting Model\n"); f.flush()
    x = x.Model
    f.write("getting Mesh\n"); f.flush()
    m = x.Mesh
    for obj in [m] + list(m.Children):
        f.write("OBJECT %s %s\n" % (obj.Name, obj.DataModelObjectCategory))
        f.flush()
        for attr in ("ElementSize", "RelevanceCenter", "GrowthRate", "UseAdaptiveSizing",
                     "CaptureCurvature", "CaptureProximity", "MaxSize", "MinSize",
                     "NumberOfDivisions", "InflationOption", "FirstLayerHeight"):
            try:
                f.write("  %s = %s\n" % (attr, getattr(obj, attr)))
            except Exception:
                pass
    f.write("MESH children = %s\n" % len(m.Children))
except Exception:
    f.write(traceback.format_exc())
f.flush()
f.close()
