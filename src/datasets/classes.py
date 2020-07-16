from typing import Dict

from src.utils import registry


@registry.register("classes", "classes")
class Classes:
    def __init__(self, path):
        self.path = path

        self.classes_dict = {}

        with open(path, "r") as in_file:
            for index, line in enumerate(in_file):
                line = line.strip()
                self.classes_dict[index + 1] = line

    def get_classes_dict(self) -> Dict:
        return self.classes_dict
