"""Separate diagnostic alternatives to the archived transformation.

These are partial, declared sensitivity changes, not a validated packet/TLS
simulator or a replacement for the historical paper experiment.
"""
import numpy as np

ARMS = ("historical", "byte_lengths", "zero_byte_rate", "combined")
EXTREMA = {"Fwd Pkt Len Max", "Fwd Pkt Len Min", "Bwd Pkt Len Max",
           "Bwd Pkt Len Min", "Pkt Len Max", "Pkt Len Min"}


def transform(X, cfg, historical_operator, arm):
    if arm not in ARMS:
        raise ValueError("Unknown sensitivity arm: " + arm)
    shifted = historical_operator(X, cfg)
    if arm == "historical":
        return shifted
    eligible = ((X["Tot Fwd Pkts"] > 0) & (X["Tot Bwd Pkts"] > 0)).to_numpy()
    values = lambda name: X[name].to_numpy(dtype=np.float64)
    allowed = set()
    if arm in ("byte_lengths", "combined"):
        # Equal allocation of the declared added bytes to the declared added
        # packets. In the principal footprint, this is one packet per direction.
        lengths = []
        for prefix, tag in (("Fwd", "fwd"), ("Bwd", "bwd")):
            packets, byte_delta = cfg[f"d_{tag}_pkts"], cfg[f"d_{tag}_bytes"]
            if packets <= 0 or packets != int(packets) or byte_delta <= 0:
                raise ValueError("Positive integer packet additions and positive bytes required.")
            length = byte_delta / packets
            if length != int(length) or length > cfg["added_pkt_size"]:
                raise ValueError("This plan requires equal integer payload lengths within its declared size bound.")
            lengths.append(length)
            for suffix, operation in (("Max", np.maximum), ("Min", np.minimum)):
                column = f"{prefix} Pkt Len {suffix}"
                shifted[column] = np.where(eligible, operation(values(column), length), values(column)).astype("float32")
        for suffix, operation, added in (("Max", np.maximum, max(lengths)),
                                         ("Min", np.minimum, min(lengths))):
            column = "Pkt Len " + suffix
            shifted[column] = np.where(eligible, operation(values(column), added), values(column)).astype("float32")
        allowed |= EXTREMA
    if arm in ("zero_byte_rate", "combined"):
        original_bytes = values("TotLen Fwd Pkts") + values("TotLen Bwd Pkts")
        original_packets = values("Tot Fwd Pkts") + values("Tot Bwd Pkts")
        zero = eligible & (original_bytes == 0)
        if np.any(zero & ((values("Flow Pkts/s") <= 0) | (values("Flow Duration") <= 0))):
            raise ValueError("Zero-byte rate alternative requires positive original duration and packet rate.")
        # R_packets=N/D implies R_bytes'=R_packets*(B'/N), with the same D.
        # This avoids choosing a duration unit and leaves positive-byte rows
        # on the original ratio-scaling branch.
        added_bytes = cfg["d_fwd_bytes"] + cfg["d_bwd_bytes"]
        rate = values("Flow Pkts/s") * np.divide(original_bytes + added_bytes,
            original_packets, out=np.zeros(len(X)), where=original_packets > 0)
        shifted.loc[zero, "Flow Byts/s"] = rate[zero].astype("float32")
        allowed.add("Flow Byts/s")
    historical = historical_operator(X, cfg)
    untouched = [c for c in X if c not in allowed]
    if not np.array_equal(shifted[untouched].to_numpy(), historical[untouched].to_numpy()):
        raise ValueError("The alternative changed a column outside its declared scope.")
    if not np.array_equal(shifted.to_numpy()[~eligible], X.to_numpy()[~eligible]):
        raise ValueError("The alternative changed an ineligible row.")
    if not np.isfinite(shifted.to_numpy()).all():
        raise ValueError("Nonfinite value in the alternative transformation.")
    return shifted
