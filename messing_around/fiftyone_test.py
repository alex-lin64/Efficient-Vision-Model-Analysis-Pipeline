import fiftyone as fo


dataset = fo.Dataset.from_dir(
    "data/",
    dataset_type = fo.types.ImageDirectory,
    name = "test_dataset2"
)
dataset.persistent = False

session = fo.launch_app(dataset, port=5151)

# samples to annotate
selected_view = dataset.select(session.selected)
# unique id for this annotation run
anno_key = "test_run"

# upload samples, run cvat
anno_results = selected_view.annotate(
    anno_key,
    label_filed="",
    label_type="detections",
    classes=['horse', 'person', 'dog', 'bike', 'car'],
    launch_editor=True
)

# print status
status = anno_results.get_status()
anno_results.print_status()




session.wait()

