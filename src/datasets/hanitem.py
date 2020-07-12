import attr


@attr.s
class HANItem:
    sentences = attr.ib()
    label = attr.ib()
