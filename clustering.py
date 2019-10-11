# -----------------------------------------------------------------------------
# Copyright (c) 2016, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

from collections import OrderedDict, deque
from six.moves import zip

import numpy as np
import pandas

__all__ = ["Clustering"]


class Clustering(object):
    # using a modified DBSCAN

    SignalDtype = np.dtype(
        [
            ("t", np.float64),
            ("r", np.float64),
            ("v", np.float64),
            ("snr", np.float64),
            ("rcs", np.float64),
            ("pulse_num", np.uint32),
            ("cluster", np.uint32),
            ("core", np.bool_),
        ]
    )

    def __init__(self, eps=0.5, min_samples=5, tscale=1, rscale=1, vscale=1):
        self.eps = eps
        self.min_samples = min_samples
        self.tscale = tscale
        self.rscale = rscale
        self.vscale = vscale
        self._tscale_sq = tscale ** 2
        self._rscale_sq = rscale ** 2
        self._vscale_sq = vscale ** 2
        self._neg_eps_tscale = -eps * tscale

        self._visited = deque()
        self._visited_dists = deque()
        self._new = deque()
        self._next_cluster = 1

        self.activeclusters = OrderedDict()

    def distance(self, p, o):
        dt = p["t"] - o["t"]
        dr = p["r"] - o["r"]
        dv = p["v"] - o["v"]
        av = 0.5 * (p["v"] + o["v"])

        dist = np.sqrt(
            dt ** 2 / self._tscale_sq
            + (dr - av * dt) ** 2 / self._rscale_sq
            + dv ** 2 / self._vscale_sq
        )
        return dist

    def addnext(self, **kwargs):
        # add new signal point with values specified by keyword arguments
        p = np.zeros(1, Clustering.SignalDtype)
        for key, value in kwargs.items():
            p[key] = value
        self._new.append(p)

        t_visit = p["t"][0] + self._neg_eps_tscale
        clusters = []
        while self._new[0]["t"][0] < t_visit:
            # we have enough subsequent points to determine if next new point
            # is a core point
            c = self._visitnew()
            if c:
                clusters.extend(c)
        return clusters

    def finish(self):
        # complete clustering when no new points will be added
        # cluster remaining new points (and write out remaining data for visited points)
        clusters = []
        while len(self._new) > 0:
            c = self._visitnew()
            if c:
                clusters.extend(c)
        return clusters

    def _visitnew(self):
        # take the first point off the new deque and visit it
        p = self._new.popleft()

        # find neighbors from visited points
        neighbors = []
        for o, odists in zip(self._visited, self._visited_dists):
            # distances to p (first point in new_points deque) are the first
            # values in odists, pop them so this is true for new point next time
            if odists.popleft() < self.eps:
                neighbors.append(o)

        # calculate distances to other new points
        try:
            newarr = np.concatenate(self._new)
        except ValueError:
            newdists = deque()
        else:
            newdists = deque(self.distance(p, newarr))

        # find neighbors from new points
        for o, odist in zip(self._new, newdists):
            if odist < self.eps:
                neighbors.append(o)

        # cluster
        if len(neighbors) >= self.min_samples:
            # p is a core point, it belongs in a cluster
            p["core"] = True

            # determine/assign cluster for p
            pclust = p["cluster"][0]
            if pclust == 0:
                # create new cluster
                pclust = self._next_cluster
                self._next_cluster += 1
                p["cluster"] = pclust

            # find clusters of neighboring core points for merging
            neighb_clusts = set()
            noncore_neighb = []
            for o in neighbors:
                if o["core"][0]:
                    if o["cluster"][0] != pclust:
                        neighb_clusts.add(o["cluster"][0])
                else:
                    noncore_neighb.append(o)
            # merge clusters, switching neighbors' cluster to pclust
            for merge_clust in neighb_clusts:
                for on in self._new:
                    if on["cluster"][0] == merge_clust:
                        on["cluster"] = pclust
                for ov in self._visited:
                    if ov["cluster"][0] == merge_clust:
                        ov["cluster"] = pclust
                if merge_clust in self.activeclusters:
                    oldrows = self.activeclusters.pop(merge_clust)
                    for orow in oldrows:
                        orow["cluster"] = pclust
                    self.activeclusters[pclust] = oldrows

            # p is core, so add all unclustered, non-core neighbors to same cluster
            for o in noncore_neighb:
                if o["cluster"][0] == 0:
                    o["cluster"] = pclust

        # add p to visited points and store its distances
        self._visited.append(p)
        self._visited_dists.append(newdists)

        # check and expire points that are too far away to be newly classified
        # (distance deque has run out of values)
        clusters = []
        while len(self._visited_dists) > 0 and len(self._visited_dists[0]) == 0:
            ret = self._expire_oldest_visited()
            if ret is not None:
                clusters.append(ret)
        return clusters

    def _expire_oldest_visited(self):
        # write data for oldest visited point to file and remove from memory
        self._visited_dists.popleft()
        o = self._visited.popleft()
        oclust = o["cluster"][0]
        if oclust != 0:
            # add point to dictionary of active clusters
            self.activeclusters.setdefault(oclust, []).append(o)
            # check to see if cluster is still active
            # (i.e. there are points in that cluster that are not expired)
            isactive = False
            for ov in self._visited:
                if ov["cluster"] == oclust:
                    isactive = True
                    break
            if not isactive:
                clusterpoints = self.activeclusters.pop(oclust)
                return pandas.DataFrame(np.concatenate(clusterpoints))
        return None
