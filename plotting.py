# -----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
from mpl_toolkits import axes_grid1

from time_utils import datetime_from_float, datetime_to_float, timestamp_strftime

__all__ = [
    "rtiplot",
    "implot",
    "add_colorbar",
    "size_dpi_nointerp",
    "make_axes_fixed",
    "rotate_ticklabels",
    "arrayticks",
    "timeticks_helper",
    "timeticks_array",
    "timeticks",
]


def rtiplot(z, t, r, **kwargs):
    kwargs["xistime"] = True
    return implot(z, t, r, **kwargs)


def implot(
    z,
    x,
    y,
    xlabel=None,
    ylabel=None,
    title=None,
    exact_ticks=True,
    xbins=10,
    ybins=10,
    xistime=False,
    yistime=False,
    cbar=True,
    clabel=None,
    cposition="right",
    csize=0.125,
    cpad=0.1,
    cbins=None,
    ax=None,
    pixelaspect=None,
    **kwargs
):
    imshowkwargs = dict(aspect="auto", interpolation="none", origin="lower")

    # asarray needed to convert pandas' DatetimeIndex to datetime64
    if xistime:
        x = np.asarray(x)
    if yistime:
        y = np.asarray(y)

    if exact_ticks:
        extent = (-0.5, x.shape[0] - 0.5, -0.5, y.shape[0] - 0.5)
    else:
        if xistime:
            # use day of the given first time as epoch
            xepoch = x[0].astype("datetime64[D]").astype(x[0].dtype)
            x_float = datetime_to_float(x, epoch=xepoch)
            xstart = x_float[0]
            xend = x_float[-1]
        else:
            xstart = x[0]
            xend = x[-1]
        xstep = (xend - xstart) / (x.shape[0] - 1)
        if yistime:
            # use day of the given first time as epoch
            yepoch = y[0].astype("datetime64[D]").astype(x[0].dtype)
            y_float = datetime_to_float(y, epoch=yepoch)
            ystart = y_float[0]
            yend = y_float[-1]
        else:
            ystart = y[0]
            yend = y[-1]
        ystep = (yend - ystart) / (y.shape[0] - 1)
        extent = (
            xstart - xstep / 2.0,
            xend + xstep / 2.0,
            ystart - ystep / 2.0,
            yend + ystep / 2.0,
        )
    imshowkwargs.update(extent=extent)

    if pixelaspect is not None:
        box_aspect = abs((extent[1] - extent[0]) / (extent[3] - extent[2]))
        arr_aspect = float(z.shape[0]) / z.shape[1]
        aspect = box_aspect / arr_aspect / pixelaspect
        imshowkwargs.update(aspect=aspect)

    imshowkwargs.update(kwargs)

    if ax is None:
        ax = plt.gca()
    img = ax.imshow(z.swapaxes(0, 1), **imshowkwargs)
    ax.grid(False)

    if cbar:
        add_colorbar(
            img,
            ax=ax,
            position=cposition,
            size=csize,
            pad=cpad,
            label=clabel,
            bins=cbins,
        )

    # title and labels
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # format and locate the ticks
    if exact_ticks:
        if xistime:
            timeticks_array(ax.xaxis, x, xbins)
        else:
            arrayticks(ax.xaxis, x, xbins)
        if yistime:
            timeticks_array(ax.yaxis, y, ybins)
        else:
            arrayticks(ax.yaxis, y, ybins)
    else:
        if xistime:
            timeticks(ax.xaxis, x[0], x[-1], xepoch, xbins)
        else:
            ax.xaxis.set_major_locator(
                mpl.ticker.MaxNLocator(nbins=xbins, integer=False)
            )
        if yistime:
            timeticks(ax.yaxis, y[0], y[-1], yepoch, ybins)
        else:
            ax.yaxis.set_major_locator(
                mpl.ticker.MaxNLocator(nbins=ybins, integer=False)
            )

    # rotate the ticks by default if they are time ticks
    if xistime:
        rotate_ticklabels(ax.xaxis, -45)
    if yistime:
        rotate_ticklabels(ax.yaxis, 45)

    return img


def add_colorbar(
    mappable,
    ax,
    position="right",
    size=0.125,
    pad=0.15,
    label=None,
    bins=None,
    **kwargs
):
    # add a colorbar that resizes with the image
    fig = ax.get_figure()
    # delete any existing colorbar
    if mappable.colorbar is not None:
        oldcb = mappable.colorbar
        oldcax = oldcb.ax
        # delete colorbar axes from figure
        fig.delaxes(oldcax)
        # restore axes to original divider
        if hasattr(mappable, "axesloc"):
            origloc = mappable.axesloc
            ax.set_axes_locator(origloc)
        # delete colorbar reference
        mappable.colorbar = None
        del oldcb, oldcax, origloc

    # save original locator as attribute (so we can delete colorbar, see above)
    origloc = ax.get_axes_locator()
    mappable.axesloc = origloc
    # make axes locatable so we can use the resulting divider to add a colorbar
    axdiv = make_axes_locatable(ax)

    # create colorbar and its axes
    cax = axdiv.append_axes(
        position, size=size, pad=pad, axes_class=axes_grid1.parasite_axes.HostAxes
    )
    if position in ("bottom", "top"):
        orientation = "horizontal"
    else:
        orientation = "vertical"
    cb = fig.colorbar(mappable, cax=cax, ax=ax, orientation=orientation, **kwargs)
    # add colorbar reference to image
    mappable.colorbar = cb
    if label is not None:
        cb.set_label(label)

    # adjust number of tick bins if desired
    if bins is not None:
        tickloc = mpl.ticker.MaxNLocator(nbins=bins, integer=False)
        cb.locator = tickloc
        # must be called whenever colorbar tick locator or formatter is changed
        cb.update_ticks()

    # make current axes ax (to make sure it is not cax)
    fig.sca(ax)

    return cb


def add_colorbar_datalim_marks(cb):
    cax = cb.ax
    cax2 = cax.twinx()
    cax.axis["right"].set_visible(True)
    cax2.axis["left"].set_visible(True)
    cax2.axis["right", "top", "bottom"].set_visible(False)
    cax2.yaxis.set_ticks_position("left")
    cax2.yaxis.set_label_position("left")
    cax2.set_ylim(cb.mappable.get_clim())
    vals = cb.mappable.get_array()
    if len(vals) > 0:
        cax2.set_yticks((np.min(vals), np.max(vals)))
        cax2.set_yticklabels([">", ">"])
    cax2.tick_params(axis="y", direction="out", length=0, pad=0)

    return cax2


def size_dpi_nointerp(nx, ny, maxwidth, maxheight):
    """width, height, dpi for displaying pixels with no interpolation"""
    xdpi_min = int(np.ceil(nx / float(maxwidth)))
    ydpi_min = int(np.ceil(ny / float(maxheight)))
    dpi = max(xdpi_min, ydpi_min)
    xstretch = dpi // xdpi_min
    ystretch = dpi // ydpi_min
    width = float(nx) / dpi * xstretch
    height = float(ny) / dpi * ystretch

    return width, height, dpi


def make_axes_fixed(ax, xinches, yinches, aspect="auto"):
    # make a fixed size divider, located using existing locator if necessary
    div = axes_grid1.axes_divider.AxesDivider(
        ax, xref=axes_grid1.Size.Fixed(xinches), yref=axes_grid1.Size.Fixed(yinches)
    )
    origloc = ax.get_axes_locator()
    if origloc is not None:
        div.set_locator(origloc)

    # place the axes in the new divider
    loc = div.new_locator(0, 0)
    ax.set_axes_locator(loc)

    # default aspect is 'auto' so the data is scaled to fill the desired size
    ax.set_aspect(aspect)

    return div


def make_axes_locatable(ax):
    # custom make_axes_locatable to fix:
    #  - case when axes is already locatable and we want to work within
    #    existing divider
    #  - case when axes has a specified aspect ratio other than 1 or auto

    origloc = ax.get_axes_locator()
    if origloc is None:
        # create axes divider that follows size of original axes (the subplot's area)

        # default AxesDivider has relative lengths in data units,
        # i.e. if the x-axis goes from x0 to x1, then the horizontal size of
        # the axes has a relative length of (x1 - x0)

        # when the axes' aspect ratio is set, however, the default axes divider
        # scales the divider size so that aspect ratio is fixed at 1 regardless
        # of the specified aspect ratio

        # in order to make aspect ratios other than 1 work, we need to scale
        # the relative length for the y-axis by the aspect ratio

        # set relative length for x-axis based on data units of ax
        hsize = axes_grid1.Size.AxesX(ax)

        # set relative length for y-axis based on aspect-scaled data units
        aspect = ax.get_aspect()
        if aspect == "equal":
            aspect = 1

        if aspect == "auto":
            vsize = axes_grid1.Size.AxesY(ax)
        else:
            vsize = axes_grid1.Size.AxesY(ax, aspect=aspect)

        div = axes_grid1.axes_divider.AxesDivider(ax, xref=hsize, yref=vsize)
    else:
        origdiv = origloc._axes_divider
        # make new axes divider (since we can't presume to modify original)
        hsize = axes_grid1.Size.AddList(
            origdiv.get_horizontal()[origloc._nx : origloc._nx1]
        )
        vsize = axes_grid1.Size.AddList(
            origdiv.get_vertical()[origloc._ny : origloc._ny1]
        )
        div = axes_grid1.axes_divider.AxesDivider(ax, xref=hsize, yref=vsize)
        div.set_aspect(origdiv.get_aspect())
        div.set_anchor(origdiv.get_anchor())
        div.set_locator(origloc)

    # place the axes in the new divider
    loc = div.new_locator(0, 0)
    ax.set_axes_locator(loc)

    return div


def arrayticks(axis, arr, nbins=10):
    def tickformatter(x, pos=None):
        try:
            idx = int(round(x))
            val = arr[idx]
        except IndexError:
            s = ""
        else:
            if isinstance(val, float):
                s = "{0:.3f}".format(val).rstrip("0").rstrip(".")
            else:
                s = str(val)
            if pos is None:
                s = s + " ({0})".format(idx)
        return s

    axis.set_major_formatter(mpl.ticker.FuncFormatter(tickformatter))
    axis.set_major_locator(mpl.ticker.MaxNLocator(nbins=nbins, integer=True))


def timeticks_helper(ts, te):
    # get common string to label time axis
    tts = ts.timetuple()
    tte = te.timetuple()
    # compare year
    if tts[0] != tte[0]:
        tlabel = ""

        def sfun(ttick):
            return (
                timestamp_strftime(ttick, "%Y-%m-%d %H:%M:%S.%f")
                .rstrip("0")
                .rstrip(".")
            )

    # compare month
    elif tts[1] != tte[1]:
        tlabel = str(tts[0])

        def sfun(ttick):
            return (
                timestamp_strftime(ttick, "%b %d, %H:%M:%S.%f").rstrip("0").rstrip(".")
            )

    # compare day of month
    elif tts[2] != tte[2]:
        tlabel = timestamp_strftime(ts, "%B %Y")

        def sfun(ttick):
            return timestamp_strftime(ttick, "%d, %H:%M:%S.%f").rstrip("0").rstrip(".")

    # compare hour
    elif tts[3] != tte[3]:
        tlabel = timestamp_strftime(ts, "%b %d %Y")

        def sfun(ttick):
            return timestamp_strftime(ttick, "%H:%M:%S.%f").rstrip("0").rstrip(".")

    # compare minute
    elif tts[4] != tte[4]:
        tlabel = timestamp_strftime(ts, "%b %d %Y, %H:xx")

        def sfun(ttick):
            return timestamp_strftime(ttick, "%M:%S.%f").rstrip("0").rstrip(".")

    # compare second
    elif tts[5] != tte[5]:
        tlabel = timestamp_strftime(ts, "%b %d %Y, %H:%M:xx (s)")

        def sfun(ttick):
            return timestamp_strftime(ttick, "%S.%f").rstrip("0").rstrip(".")

    else:
        tlabel = timestamp_strftime(ts, "%b %d %Y, %H:%M:%S+ (s)")

        def sfun(ttick):
            return timestamp_strftime(ttick, "0.%f")

    return tlabel, sfun


def timeticks_array(axis, arr, nbins=10):
    # convert time array to pandas DatetimeIndex,
    # which returns Timestamp objects when indexed
    arr_idx = pandas.DatetimeIndex(arr)

    tlabel, sfun = timeticks_helper(arr_idx[0], arr_idx[-1])
    currlabel = axis.get_label_text()
    if currlabel != "":
        tlabel = tlabel + "\n" + currlabel
    axis.set_label_text(tlabel)

    def tickformatter(x, pos=None):
        idx = int(round(x))
        try:
            val = arr_idx[idx]
        except IndexError:
            s = ""
        else:
            s = sfun(val)
            if pos is None:
                s = s + " ({0})".format(idx)
        return s

    axis.set_major_formatter(mpl.ticker.FuncFormatter(tickformatter))
    axis.set_major_locator(mpl.ticker.MaxNLocator(nbins=nbins, integer=True))


def timeticks(axis, ts, te, floatepoch, nbins=10):
    # convert ts and te to Timestamp objects
    ts = pandas.Timestamp(ts)
    te = pandas.Timestamp(te)

    tlabel, sfun = timeticks_helper(ts, te)
    currlabel = axis.get_label_text()
    if currlabel != "":
        tlabel = tlabel + "\n" + currlabel
    axis.set_label_text(tlabel)

    def tickformatter(x, pos=None):
        ttick = pandas.Timestamp(datetime_from_float(x, "ns", epoch=floatepoch).item())
        s = sfun(ttick)
        return s

    axis.set_major_formatter(mpl.ticker.FuncFormatter(tickformatter))
    axis.set_major_locator(mpl.ticker.MaxNLocator(nbins=nbins, integer=False))


def rotate_ticklabels(axis, rotation=45, minor=False):
    """Rotate ticklabels for the given axis.

    Based on the tick position and the rotation angle, the labels will be aligned
    so that they line up nicely with the ticks.

    """
    if minor:
        ticks = axis.get_minor_ticks()
    else:
        ticks = axis.get_major_ticks()

    if isinstance(axis, mpl.axis.XAxis):
        poses = ["bottom", "top"]
    elif isinstance(axis, mpl.axis.YAxis):
        poses = ["left", "right"]

    # get tick labels for bottom/top or left/right
    labels1 = [tick.label1 for tick in ticks]
    labels2 = [tick.label2 for tick in ticks]

    labelses = [labels1, labels2]

    for labels, pos in zip(labelses, poses):
        for label in labels:
            label.set_rotation(rotation)
            # anchor makes it possible to "center" end of labels on tick
            label.set_rotation_mode("anchor")

            # rotation wrapped to [0, 360]
            rot = label.get_rotation()

            if pos == "left":
                if rot <= 180:
                    label.set_va("center")
                else:
                    label.set_va("top")
            elif pos == "right":
                if rot <= 180:
                    label.set_va("top")
                else:
                    label.set_va("center")
            elif pos == "bottom":
                if rot == 0:
                    label.set_ha("center")
                elif rot <= 180:
                    label.set_ha("right")
                else:
                    label.set_ha("left")
            elif pos == "top":
                label.set_va("baseline")
                if rot == 0:
                    label.set_ha("center")
                elif rot <= 180:
                    label.set_ha("left")
                else:
                    label.set_ha("right")
