# This script is intended for application deployment.
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from cestel_helpers.version import generate_version_file

# This is current convention, but if someone comes up with better ideas, I'm listening.
version_file_name = '.version'

# List of files which will be added to the ZIP and for which hashes are calculated and stored in the version file.
files = ['README.md', 'main.py']

ver = generate_version_file(version_file_name, files=files)

zip_name = 'template_{}.zip'.format('.'.join(ver))

Path('releases').mkdir(exist_ok=True)
with ZipFile(Path('releases', zip_name), 'w') as zip_file:
    for f in files + [version_file_name]:
        zip_file.write(f, compress_type=ZIP_DEFLATED)
