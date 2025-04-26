from app.services.dataset_service import DatasetService

def test_add_and_list():
    service = DatasetService()
    item = {"name": "test", "desc": "for test"}
    service.add_dataset(item)
    datasets = service.list_datasets()
    assert any(d["name"] == "test" for d in datasets) 