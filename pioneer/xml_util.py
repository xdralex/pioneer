from typing import Optional
from xml.etree.ElementTree import Element


def find_unique(parent: Element, tag_or_path: str, attr_name: Optional[str] = None, attr_value: Optional[str] = None) -> Element:
    if attr_name is None:
        results = parent.findall(tag_or_path)
    else:
        results = [x for x in parent.findall(tag_or_path) if x.attrib.get(attr_name) == attr_value]

    assert len(results) == 1
    return results[0]
