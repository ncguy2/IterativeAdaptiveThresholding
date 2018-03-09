def CeilPos(n):
    """
    Rounds up the provided number
    :param n: The number to round, Must be positive
    :return: The result of math.ceil(n)
    """
    if n - int(n) > 0:
        return int(n + 1)
    return int(n)


def CeilNeg(n):
    """
    Rounds up the provided number
    :param n: The number to round, Must be negative
    :return: The result of math.ceil(n)
    """
    if n - int(n) < 0:
        return int(n - 1)
    return int(n)


def Ceil(n):
    """
    Local implementation of math.ceil to achieve zero dependencies
    :param n: The value to ceil
    :return: the result of math.ceil(n)
    """
    if n > 0:
        return CeilPos(n)
    return CeilNeg(n)


class IAT_Cell(object):
    """
    Base cell for the internal quad-tree structure
    """

    def __init__(self, root, cellType: str, depth=0):
        """
        :param root: The IAT object this cell belongs to
        :param cellType: A string denoting which cell type this is
        :param depth: Internal counter for the depth within the quad-tree
        """
        self.root = root
        self.type = cellType
        self.depth = depth

    def Draw(self, frame, drawFunc, depth=0):
        """
        Draws the cell to the provided image using the provide draw function
        :param frame: The image container to draw to
        :param drawFunc: The draw function provided by the root object
        :param depth: The internal counter of the cell
        """
        raise NotImplementedError("Please Implement this method")

    def Flatten(self, items):
        """
        Flattens all cells and returns an array of IAT_Item objects
        :param items: The item tree to flatten
        :return: All leaf nodes in the provided sub-tree
        """
        raise NotImplementedError("Please Implement this method")

    def ToDebugImage(self, img):
        """
        Draws the cell threshold masks to the provided image container
        :param img: The image container to write to
        """
        raise NotImplementedError("Please Implement this method")

    def ToImage(self, img):
        """
        Draws the cell states to the provided image container
        :param img: The image container to write to
        """
        raise NotImplementedError("Please Implement this method")

    def ToBinaryImage(self, img, desiredDepth, colours=None):
        """
        Draws the cell depth to the provided image container
        :param img: The image container to write to
        :param desiredDepth: How deep the render traversal should go
        :param colours: The colours to use when rendering the depth layers, defaults to solid white
        """
        raise NotImplementedError("Please Implement this method")


class IAT_Node(IAT_Cell):
    """
    Quad-tree node containing 4 IAT_Cell object, and accessors for each
    """

    topLeftIdx = 0
    topRightIdx = 1
    bottomLeftIdx = 2
    bottomRightIdx = 3

    def __init__(self, root, items=None, depth=0):
        """
        :param root: The IAT object this cell belongs to
        :param items: Array of IAT_Item objects to be used as the children
        :param depth: Internal counter for the depth within the quad-tree
        """
        if items is None:
            items = []

        IAT_Cell.__init__(self, root, "NODE", depth)
        self.items = items

        for i in range(len(self.items)):
            self.items[i].parent = self
            self.items[i].slot = i

    def GetTopLeft(self):
        """
        :return: The top left cell
        """
        return self.Get(self.topLeftIdx)

    def GetTopRight(self):
        """
        :return: The top right cell
        """
        return self.Get(self.topRightIdx)

    def GetBottomLeft(self):
        """
        :return: The bottom left cell
        """
        return self.Get(self.bottomLeftIdx)

    def GetBottomRight(self):
        """
        :return: The bottom right cell
        """
        return self.Get(self.bottomRightIdx)

    def Set(self, idx, item):
        """
        Sets the contents of the slot at the provided index
        :param idx: The slot index
        :param item: The item to put in the slot
        """
        self.items[idx] = item

    def Get(self, idx):
        """
        Gets the contents of the slot at the provided index
        :param idx: The slot index
        :return: The contents of the slot at the provided index
        """
        return self.items[idx]

    def Split(self, idx):
        """
        Splits the item in the slot at the provided index
        :param idx: The slot index
        """
        self.Get(idx).Split()

    def Draw(self, frame, drawFunc, depth=0):
        """
        Draws the contained child cells to the provided image using the provide draw function
        :param frame: The image container to draw to
        :param drawFunc: The draw function provided by the root object
        :param depth: The internal counter of the cell
        """
        for i in range(len(self.items)):
            self.items[i].Draw(frame, drawFunc, depth + 1)

    def Flatten(self, items):
        """
        Flattens all children cells and returns an array of IAT_Item objects
        :param items: The item tree to flatten
        :return: All leaf nodes in the provided sub-tree
        """
        for i in range(len(self.items)):
            self.items[i].Flatten(items)

    def ToDebugImage(self, img):
        """
        Draws the contained children cells threshold masks to the provided image container
        :param img: The image container to write to
        """
        for i in range(len(self.items)):
            self.items[i].ToDebugImage(img)

    def ToImage(self, img):
        """
        Draws the contained children cell states to the provided image container
        :param img: The image container to write to
        """
        for i in range(len(self.items)):
            self.items[i].ToImage(img)

    def ToBinaryImage(self, img, desiredDepth, colours=None):
        """
        Draws the contained children cell depth to the provided image container
        :param img: The image container to write to
        :param desiredDepth: How deep the render traversal should go
        :param colours: The colours to use when rendering the depth layers, defaults to solid white
        """
        for i in range(len(self.items)):
            self.items[i].ToBinaryImage(img, desiredDepth, colours)


class IAT_Item(IAT_Cell):
    """
    Leaf node of the IAT quad-tree
    Contains the ROI coordinates and is capable of splitting itself in 4
    """

    def __init__(self, root, img, rect, depth, parent=None, slot=None):
        """
        :param root: The IAT object this cell belongs to
        :param img: The whole image container
        :param rect: The ROI coordinates of the item
        :param depth: Internal counter for the depth within the quad-tree
        :param parent: The node parent object
        :param slot: The slot index within the parent object
        """
        IAT_Cell.__init__(self, root, "ITEM", depth)
        self.image = img
        self.rect = rect
        self.roi = img[rect[0]:rect[0] + rect[2], rect[1]:rect[1] + rect[3]]
        self.parent = parent
        self.slot = slot
        self.filled = False

    def Split(self) -> IAT_Node:
        """
        Splits the cell into 4 smaller cells, separating the ROI equally among the cells
        Will automatically change the contents of the parent slot to the new node
        :return: Returns the IAT_Node containing the new child cells
        """
        x = int(self.rect[0])
        y = int(self.rect[1])
        halfWidth = int(Ceil(self.rect[2] * .5))
        halfHeight = int(Ceil(self.rect[3] * .5))

        tl = IAT_Item(self.root, self.image, [x, y, halfWidth, halfHeight], depth=self.depth + 1)
        tr = IAT_Item(self.root, self.image, [x + halfWidth, y, halfWidth, halfHeight], depth=self.depth + 1)
        bl = IAT_Item(self.root, self.image, [x, y + halfHeight, halfWidth, halfHeight], depth=self.depth + 1)
        br = IAT_Item(self.root, self.image, [x + halfWidth, y + halfHeight, halfWidth, halfHeight],
                      depth=self.depth + 1)

        newNode = IAT_Node(self.root, [tl, tr, bl, br], depth=self.depth)

        if self.parent is not None and self.slot is not None:
            self.parent.Set(self.slot, newNode)

        return newNode

    def ShouldSplit(self) -> bool:
        """
        :return: Checks whether this item should split, based on the minimum allowed resolution, and the threhold check
        """
        if self.rect[2] * .5 <= self.root.minRes and self.rect[3] * .5 <= self.root.minRes:
            return False

        thresh = self.root.threshFunc(self.roi)
        return thresh < self.root.splitThreshold

    def Draw(self, frame, drawFunc, depth=0):
        """
        Draws the cell to the provided image using the provide draw function
        :param frame: The image container to draw to
        :param drawFunc: The draw function provided by the root object
        :param depth: The internal counter of the cell
        """
        drawFunc(frame, self.rect, depth)

    def Flatten(self, items):
        """
        Flattens all cells and returns an array of IAT_Item objects
        :param items: The item tree to flatten
        :return: All leaf nodes in the provided sub-tree
        """
        items.append(self)

    def ToDebugImage(self, img):
        """
        Draws the cell threshold masks to the provided image container
        :param img: The image container to write to
        """
        img[self.rect[0]:self.rect[0] + self.rect[2], self.rect[1]:self.rect[1] + self.rect[3]] = self.root.threshFunc(
            self.roi, True)

    def ToImage(self, img):
        """
        Draws the cell states to the provided image container
        :param img: The image container to write to
        """
        img[self.rect[0]:self.rect[0] + self.rect[2],
        self.rect[1]:self.rect[1] + self.rect[3] + 1] = self.root.threshFunc(self.roi) * 255

    def ToBinaryImage(self, img, desiredDepth, colours=None):
        """
        Draws the cell depth to the provided image container
        :param img: The image container to write to
        :param desiredDepth: How deep the render traversal should go
        :param colours: The colours to use when rendering the depth layers, defaults to solid white
        """
        show = self.depth <= desiredDepth or self.root.threshFunc(self.roi) < self.root.splitThreshold

        if show:
            if colours:
                col = colours[self.depth % len(colours)]
            else:
                col = (255, 255, 255)
        else:
            col = (0, 0, 0)

        img[self.rect[0]:self.rect[0] + self.rect[2], self.rect[1]:self.rect[1] + self.rect[3] + 1] = col


class IAT(object):
    """
    The root of the IAT structure
    """
    def __init__(self, img, minRes=16, splitThreshold=0.5):
        """
        :param img: The whole image container to measure
        :param minRes: The minimum allowed resolution of a single cell
        :param splitThreshold: The percentage that must be exceeded by the threshold result to allow for a cell to be split
        """
        self.image = img
        self.minRes = minRes
        self.imgWidth = img.shape[0]
        self.imgHeight = img.shape[1]
        self.drawFunc = False
        self.threshFunc = False
        self.currIdx = None
        self.currItems = None
        self.splitThreshold = splitThreshold
        self.item = IAT_Item(self, self.image, (0, 0, self.imgWidth, self.imgHeight), depth=0).Split()
        self.Invalidate()

    def Invalidate(self):
        """
        Invalidates the current work-group and prepares a new one
        """
        self.currIdx = 0
        self.currItems = self.Flatten()

    def HasFinishedCycle(self):
        """
        :return: Whether the work-group cycle has finished
        """
        return self.currIdx >= len(self.currItems)

    def StepAll(self, depth=1, force=False):
        """
        Completes the current work-group, repeats for the defined depth
        :param depth: The amount of work-group to complete
        :param force: Whether to force the cells to split
        """
        for i in range(depth):
            self.Invalidate()
            while not self.HasFinishedCycle():
                self.StepItem(self.currIdx, force)
                self.currIdx += 1

    def Step(self):
        """
        Executes the next step in the work-group
        """
        if self.HasFinishedCycle():
            self.Invalidate()
        self.StepItem(self.currIdx)
        self.currIdx += 1

    def StepItem(self, idx, force=False):
        """
        Steps the item at the provided index
        :param idx: The item index
        :param force: Whether to force a split
        """
        item = self.currItems[idx]
        if force or item.ShouldSplit():
            item.Split()

    def SetDrawFunc(self, drawFunc):
        """
        Sets the function to use for drawing
        :param drawFunc: Function reference signature: (Image, Rectangle, Depth)
        """
        self.drawFunc = drawFunc

    def SetThreshFunc(self, threshFunc):
        """
        Sets the function to use for thresholding
        :param threshFunc: Function reference signature: (Image)
        """
        self.threshFunc = threshFunc

    def DrawItems(self, frame):
        """
        Draws the tree to the provided image container
        :param frame: The image container to write to
        """
        if not self.drawFunc:
            return

        self.item.Draw(frame, self.drawFunc)

    def Flatten(self) -> []:
        """
        Flattens the tree and returns an array of IAT_Item objects
        :return: Array of IAT_Item objects
        """
        items = []
        self.item.Flatten(items)
        return items

    def GetImageSize(self):
        """
        :return: The stored image size
        """
        return [self.imgWidth, self.imgHeight]

    def ToDebugImage(self, img):
        """
        Draws the debug information of the tree to the provided image container
        :param img: The image container to write to
        """
        self.item.ToDebugImage(img)

    def ToImage(self, img):
        """
        Draws the average colour of each cell in the tree to the provided image container
        :param img: The image container to write to
        """
        self.item.ToImage(img)

    def ToBinaryImage(self, img, desiredDepth, colours=None):
        """
        Draws the debug information of the tree to the provided image container
        :param img: The image container to write to
        :param desiredDepth: How far down the tree to draw
        :param colours: The colours to draw the depth levels as, defaults to (255, 255, 255)
        """
        self.item.ToBinaryImage(img, desiredDepth, colours)
