"""Does the SB-context prompt reach the brachistochrone, and is the commit clean?"""
import answerer

full = ("What shape is formed when light passes through an object where each "
        "infinitesimally thin layer of the object has an index of refraction that "
        "varies with the sine of the angle that light approaches it divided by the "
        "speed of the light at that layer?")
p95 = full[:int(len(full) * 0.95)]

for label, stem in [("95% prefix", p95), ("full", full)]:
    reasoning, ans = answerer.anticipate_sa_verbose(stem, "PHYSICS")
    print(f"[{label}] commits: {ans!r}")
    print(f"   reasoning: {reasoning[:130]}")
    print(f"   terse anticipate_sa: {answerer.anticipate_sa(stem, 'PHYSICS')[0]!r}\n")
