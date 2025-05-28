"""
this module contains the classes for CV non-locality and contextuality evaluation
CV_Bell_inequality_LP: Linear optimization program for Bell inequalities based on histograms
CV_Bell_inequality_moments: Moment based SDP for Bell inequalities
CV_Bell_Ineq_state: Class for evaluating Bell inequalities for a given state. It inherits from the CV_Bell_inequality_moments class 
and from the CV_Bell_inequality_LP class, depending on the method used.

"""

from cv_sdp.utils.CV_Bell_ineq import (
    CV_Bell_Ineq_state,
    CV_Bell_inequality_moments,
    CV_Bell_inequality_LP,
)