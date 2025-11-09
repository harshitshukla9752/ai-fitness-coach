import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle between three 3D points.
    'b' is the vertex of the angle.
    """
    a = np.array(a) # First point
    b = np.array(b) # Mid point (vertex)
    c = np.array(c) # End point
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate dot product and magnitudes
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-7
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc + epsilon)
    
    # Clip value to be between -1 and 1 to avoid acos domain errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg