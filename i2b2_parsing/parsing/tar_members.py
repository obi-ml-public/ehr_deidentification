from pathlib import Path


class TARMembers(object):
    """
    This class is used to read the the tar file and
    get the names of the files (ending with a specific
    suffix) compressed into the tar file.
    """

    @staticmethod
    def get_members(tar_obj, suffix):
        """
        Read the the tar file and
        get the names of the files (ending with a specific
        suffix) compressed into the tar file.
        Args:
            tar_obj (tarfile.TarFile): A tar object
            suffix (str): The suffix filter. Only compressed files ending with
                          this suffix will be returned
        Returns:
            (list): List containing the names of the compressed files ending with the
                    the specified suffix
        """
        return [obj.name for obj in tar_obj.getmembers() if Path(obj.name).suffix == suffix]
