from PIL import Image


class PatchExtractor:
    def __init__(self, path, patch_size, stride):
        '''
        :param path: path to the image file
        :param patch_size: integer, size of the patch
        :param stride: integer, size of the stride
        '''
        self.path = path
        self.size = patch_size
        self.stride = stride

    def extract_patches(self):
        """
        extracts all patches from an image
        :returns: A list of :py:class:`~PIL.Image.Image` objects.
        """
        with Image.open(self.path) as img:
            wp = int((img.width - self.size) / self.stride + 1)
            hp = int((img.height - self.size) / self.stride + 1)
            return [self._extract_patch(img, (w, h)) for h in range(hp) for w in range(wp)]

    def extract_patch(self, patch):
        """
        extracts a patch from an input image
        :param patch: a tuple
        :rtype: :py:class:`~PIL.Image.Image`
        :returns: An :py:class:`~PIL.Image.Image` object.
        """
        with Image.open(self.path) as img:
            return self._extract_patch(img, patch)

    def _extract_patch(self, img, patch):
        return img.crop((
            patch[0] * self.stride,  # left
            patch[1] * self.stride,  # up
            patch[0] * self.stride + self.size,  # right
            patch[1] * self.stride + self.size  # down
        ))
