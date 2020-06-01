#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This generic helping side code provides convenient functions and classes to
    every other module's "Run As A Script" section.
"""

###############################################################################
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
###############################################################################
import pathlib
import numpy as np
import scipy.io


###############################################################################
# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------
###############################################################################
class RunAsScriptStore(object):
    """
    A class for storing Run-As-Script-defined variables while maintaining
        a clean namespace.
    It provides easy savemat and loadmat operations to all self's attributes.
    """
    def savemat(self, file_name, mdict=None, appendmat=True, **kwargs):
        """
        Call scipy.io.savemat on RunAsScript attributes.
        ------------------------------------------------------------------------
        DESCRIPTION COPIED FROM scipy.io.savemat !
        Parameters
        ----------
        file_name : str or file-like object
            Name of the .mat file (.mat extension not needed if ``appendmat ==
            True``).
            Can also pass open file_like object.
        mdict : dict
            Dictionary from which to save matfile variables.
        appendmat : bool, optional
            True (the default) to append the .mat extension to the end of the
            given filename, if not already present.
        kwargs : dict
           Other keyword arguments of scipy.io.savemat.
        """
        # Set mdict default value to empty dictionary
        if mdict is None:
            mdict = {}

        # Merge mdict with attributes dictionary, giving mdict the upper-hand
        #   in case of inconsistency
        dsavemat = {**vars(self), **mdict}

        # Save the merged dictionary to a .mat file
        scipy.io.savemat(file_name, dsavemat, appendmat, **kwargs)

    def loadmat(self, file_name, mdict=None, appendmat=True, **kwargs):
        """
        Call scipy.io.loadmat into RunAsScript attributes.
        ------------------------------------------------------------------------
        DESCRIPTION COPIED FROM scipy.io.loadmat !
        Parameters
        ----------
        file_name : str
           Name of the mat file (do not need .mat extension if
           appendmat==True). Can also pass open file-like object.
        mdict : dict, optional
            Dictionary in which to insert matfile variables.
        appendmat : bool, optional
           True to append the .mat extension to the end of the given
           filename, if not already present.
        kwargs : dict
           Other keyword arguments of scipy.io.loadmat.
        Returns
        -------
        mat_dict : dict
           dictionary with variable names as keys, and loaded matrices as
           values.
        Notes
        -----
        v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.
        You will need an HDF5 python library to read MATLAB 7.3 format mat
        files.  Because scipy does not supply one, we do not implement the
        HDF5 / 7.3 interface here.
        """
        # Merge mdict with attributes dictionary, giving mdict the upper-hand
        #   in case of inconsistency
        dloadmat = scipy.io.loadmat(file_name, mdict, appendmat, **kwargs)

        # Squeeze all 2d results to their true dimension
        dloadmat = {key: np.squeeze(value) for key, value in dloadmat.items()}

        # Pour the dictionary from the .mat file into the objects fields
        self.__dict__.update(dloadmat)

        return dloadmat


###############################################################################
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
###############################################################################
# # # # # # # # # # # # # # File management functions # # # # # # # # # # # # #
def list_files_in_subfolders(fpath, pattern=r"*"):
    """List all files in folder and sub-folders that match a given pattern"""
    return tuple(pathlib.Path(fpath).rglob(pattern))