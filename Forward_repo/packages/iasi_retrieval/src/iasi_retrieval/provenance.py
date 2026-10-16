"""Immutable source identifiers for the final retrieval extraction.

These are Git blob IDs from the committed ``model`` tree. The working tree had
uncommitted z36/VMR and other later variants when this package was extracted;
those edits are intentionally not source material for this z24/Ncol recipe.
"""

SOURCE_COMMIT = "9f3e0a440189e54dd5c7ad6adfc56e74f2b04de7"
SOURCE_PREFIX = "physinv/closure_forward/experimentos/apae_loss_fisica"
SOURCE_BLOBS = {
    "_common.py": "4d279dac3db563932d734ed55b68768b37f0e2d8",
    "nets.py": "2ead4b3e76fc5ea67d57bbb6e762e38affe88572",
    "prior_z.py": "3d51fce94f846d099423ed7863150cef6cad6fdb",
    "train_apae.py": "a796496406603b8b4c2fc5f9563ff0c11c6371dc",
    "exp18_unrolled.py": "66a61a1c9b5482a1a4e20ec64a2265f2bc137a27",
    "exp16_gn_refino.py": "7c0c8317e2d961c8d29bea9f11b9d31f3be353f5",
    "exp1_barrido.py": "9912209be8923e5028820d6663d4075bd967dbca",
    "cadena_sew714.sh": "2f796fb74b5f22a50848b6c9ea78fda434766633",
}
