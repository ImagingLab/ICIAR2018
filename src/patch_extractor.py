class PatchExtractor:
    def __init__(self, img, patch_size, stride):
        '''
        :param img: :py:class:`~PIL.Image.Image`
        :param patch_size: integer, size of the patch
        :param stride: integer, size of the stride
        '''
        self.img = img
        self.size = patch_size
        self.stride = stride

    def extract_patches(self):
        """
        extracts all patches from an image
        :returns: A list of :py:class:`~PIL.Image.Image` objects.
        """
        wp, hp = self.shape()
        return [self.extract_patch((w, h)) for h in range(hp) for w in range(wp)]

    def extract_patch(self, patch):
        """
        extracts a patch from an input image
        :param patch: a tuple
        :rtype: :py:class:`~PIL.Image.Image`
        :returns: An :py:class:`~PIL.Image.Image` object.
        """
        return self.img.crop((
            patch[0] * self.stride,  # left
            patch[1] * self.stride,  # up
            patch[0] * self.stride + self.size,  # right
            patch[1] * self.stride + self.size  # down
        ))

    def shape(self):
        wp = int((self.img.width - self.size) / self.stride + 1)
        hp = int((self.img.height - self.size) / self.stride + 1)
        return wp, hp


