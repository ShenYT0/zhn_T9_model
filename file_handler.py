from pathlib import Path
from xml.etree import ElementTree as ET


class FileHandler:
    @staticmethod
    def list_xml_files(directory: str | Path, extension: str = ".xml") -> list[str]:
        """List all files in the specified directory with the given extension."""
        directory = Path(directory)
        return [f.name for f in directory.iterdir() if f.name.endswith(extension)]
