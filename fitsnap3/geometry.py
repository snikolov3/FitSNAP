# <!----------------BEGIN-HEADER------------------------------------>
# ## FitSNAP3
# A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package
#
# _Copyright (2016) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software. This software is distributed under the GNU General Public License_
# ##
#
# #### Original author:
#     Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
#     http://www.cs.sandia.gov/~athomps
#
# #### Key contributors (alphabetical):
#     Mary Alice Cusentino (Sandia National Labs)
#     Nicholas Lubbers (Los Alamos National Lab)
#     Adam Stephens (Sandia National Labs)
#     Mitchell Wood (Sandia National Labs)
#
# #### Additional authors (alphabetical):
#     Elizabeth Decolvenaere (D. E. Shaw Research)
#     Stan Moore (Sandia National Labs)
#     Steve Plimpton (Sandia National Labs)
#     Gary Saavedra (Sandia National Labs)
#     Peter Schultz (Sandia National Labs)
#     Laura Swiler (Sandia National Labs)
#
# <!-----------------END-HEADER------------------------------------->

import numpy as np

def units_conv(styles,bispec_options):
    units_conv = {}
    units_conv["Energy"] = 1.0
    units_conv["Force"] = 1.0
    units_conv["Stress"] = 1.0
    units_conv["Distance"] = 1.0

    if (bispec_options["units"]=="metal" and (list(styles["Stress"])[0]=="kbar" or list(styles["Stress"])[0]=="kB")):
        units_conv["Stress"] = 1000.0
#   Append other exceptions to unit conversion here
#        if bispec_options["verbosity"]:
#            print("Energy, Force, Stress, Distance")
#            print("From JSON: ")
#            print(list(styles["Energy"])[0],", ", list(styles["Forces"])[0],", ", list(styles["Stress"])[0],", ", list(styles["Positions"])[0])
    return units_conv

def rotate_coords(data,units_conv):
    # Transpose here because Lammps stores lattice vectors as columns,
    # QM stores lattice vectors as rows; After transposing lattice vectors are columns
    in_cell = np.asarray(data["QMLattice"]).T
    assert np.linalg.det(in_cell) > 0, "Input cell is not right-handed!"

    # Q matrix of QR decomposition is an orthogonal (rotation-like)
    # matrix whose inverse/transpose makes the input cell upper-diagonal:
    # input cell C = Q C';
    # runlammps-normalized cell C' = Q^T C.
    qmat, rmat = np.linalg.qr(in_cell)

    # Normalize signs of Q matrix to ensure positive diagonals of transformed cell;
    # QR decomposition algorithms don't always return a proper rotation
    ss = np.diagflat(np.sign(np.diag(rmat)))
    rot = ss @ qmat.T
    assert np.allclose(rot @ rot.T, np.eye(3)), "Rotation matrix not orthogonal"
    assert np.allclose(rot.T @ rot, np.eye(3)), "Rotation matrix not orthogonal"
    assert np.linalg.det(rot) > 0, "Rotation matrix is an improper rotation (det<0)"

    # ????
    # Cell transforms on first axis due to runlammps sotring lattice vectors as columns
    out_cell = rot @ in_cell

    # This assert is technically overkill, but checks that the new cell is right-handed
    assert np.linalg.det(out_cell) > 0, "New cell is not right-handed!"

    lower_triangle = out_cell[np.tril_indices(3, k=-1)]
    assert np.allclose(lower_triangle, 0, atol=1e-13), \
        f"Lower triangle of normalized cell has nonzero-elements: {lower_triangle}"
    # Positions and forces transform on the second axis
    # Stress transforms on both the first and second axis.

#    in_cell = array(config['Lattice']).transpose()
    # Inverse cell is [h k l]^t/V
    in_cellinv = np.linalg.inv(in_cell)
    # This is the LAMMPS cell
    cell = lammps_cell(in_cell)
    cell_flip(cell)
    cellprod = np.dot(cell,in_cellinv)
    # print("Rotation ",cellprod)
    # print("Orig ",data["Stress"]*units_conv["Stress"])
    # print("Mod ",np.dot(np.dot(cellprod,data["Stress"]*units_conv["Stress"]),cellprod.T))
    return {
        "Lattice": cell,
        "Positions": np.dot(cellprod,(data["Positions"]*units_conv["Distance"]).T).T,
        "Forces": np.dot(cellprod,(data["Forces"]*units_conv["Force"]).T).T,
        "Stress": np.dot(np.dot(cellprod,data["Stress"]*units_conv["Stress"]),cellprod.T),
        "Rotation": cellprod,
    }

    # return {
    #     "Lattice": out_cell,
    #     "Positions": data["Positions"]*units_conv["Distance"] @ rot.T,
    #     "Forces": data["Forces"]*units_conv["Force"] @ rot.T,
    #     "Stress": rot @ (data["Stress"]*units_conv["Stress"]) @ rot.T,
    #     "Rotation": rot,
    # }
def cell_flip(cell):

    # Check that yz is not too large for LAMMPS

    if np.abs(cell[1][2]) > 0.5*cell[1][1]:
        if cell[1][2] < 0.0:
            cell[1][2] += cell[1][1];
            cell[0][2] += cell[0][1];
        else:
            cell[1][2] -= cell[1][1];
            cell[0][2] -= cell[0][1];

    # Check that xz is not too large for LAMMPS

    if np.abs(cell[0][2]) > 0.5*cell[0][0]:
        if cell[0][2] < 0.0:
            cell[0][2] += cell[0][0];
        else:
            cell[0][2] -= cell[0][0];

    # Check that xy is not too large for LAMMPS

    if np.abs(cell[0][1]) > 0.5*cell[0][0]:
        if cell[0][1] < 0.0:
            cell[0][1] += cell[0][0];
        else:
            cell[0][1] -= cell[0][0];

    return cell

def lammps_cell(cellqm):

    cell = np.zeros((3,3))

    # Compute edge lengths

    cellqmtrans = cellqm.T
    avec = cellqmtrans[0]
    bvec = cellqmtrans[1]
    cvec = cellqmtrans[2]
    anorm = np.sqrt((avec**2.0).sum())
    bnorm = np.sqrt((bvec**2.0).sum())
    cnorm = np.sqrt((cvec**2.0).sum())
    ahat = avec/anorm

    # Inverse cell is [h k l]^t/V

    cellqminv = np.linalg.inv(cellqm)

    lvec = cellqminv[2]
    lnorm = np.sqrt((lvec**2.0).sum())
    lhat = lvec*(1.0/lnorm)

    # ax = |A|

    cell[0][0] = anorm

    # bx = |A|.|B|/|A|
    # by = Sqrt(|B|^2 - bx^2)

    cell[0][1] = np.dot(ahat,bvec)
    cell[1][1] = np.sqrt(bnorm**2 - cell[0][1]**2)

    # cx = |A|.|C|/|A|
    # cy = (|B||C| - bx*cx)/by
    # cz = Sqrt(C^2 - cx^2 - cy^2 +cz^2)

    cell[0][2] = np.dot(ahat,cvec)
    cell[1][2] = (np.dot(bvec,cvec) - cell[0][1]*cell[0][2])/cell[1][1]
    cell[2][2] = np.sqrt(cnorm**2 - cell[0][2]**2 - cell[1][2]**2)

    return cell

def translate_coords(data,units_conv):
    cell = data["Lattice"]
    position_in = data["Positions"]

    # Extra transposes because runlammps uses cells with latttice vectors as columns
    invcell = np.linalg.inv(cell.T).T
    # Fractional coordinates
    frac_coords = position_in @ invcell.T

    # Fix some rounding difficulties in divmod when within machine epsilon of zero
    frac_coords[np.isclose(frac_coords, 0, atol=1e-15)] = 0.

    trans_nums, cell_frac_coords = np.divmod(frac_coords, 1)

    assert (cell_frac_coords < 1).all(), "Fractional coordinates outside cell"
    assert (cell_frac_coords >= 0).all(), "Fractional coordinates outside cell"

    # If no translations are needed, return unmodified positions
    if (trans_nums == 0).all():
        return {
            "Positions":position_in,
            "Translation":np.zeros_like(position_in,dtype=float),
        }

    new_pos = cell_frac_coords @ cell.T
    trans_vec = trans_nums @ cell.T
    assert np.allclose(new_pos + trans_vec, position_in), "Translation failed to invert"
    return {
        "Positions": new_pos,
        "Translation": trans_vec,
    }

def check_coords(cell, pos1, pos2):
    """Compares position 1 and position 2 with respect to periodic boundaries defined by cell"""
    invcell = np.linalg.inv(np.asarray(cell).T).T

    # Fractional coordinates
    frac_1 = pos1 @ invcell.T
    frac_2 = pos2 @ invcell.T
    diff_frac = frac_2 - frac_1

    # Assert that diff_frac is very close to an integer
    assert np.allclose(
        diff_frac,
        np.round(diff_frac),
        atol=1e-12, rtol=1e-12), "Coordinates are not close after shift."+\
        "Fractional coordinate Error:{}\nArray:\n{}".format(
        np.abs(diff_frac).max(),diff_frac)
    return True

def check_volume(lattice,volume):
    assert np.allclose(np.linalg.det(lattice),volume,rtol=1e-10,atol=1e-10),\
        "Cell volume not equal to supplied volume!"
