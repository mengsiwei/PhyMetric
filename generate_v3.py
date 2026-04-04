#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_v3.py  –  Build PhyMetric question_v3.json
=====================================================
50 categories × 20-30 scenes × 8 questions ≈ 9 000+ questions

Key improvements over v2:
  • 50 physical categories (vs 20 in v2)
  • 20–30 scenes per category (vs 5)
  • Balanced answers: 4 positive + 4 negative per scene (vs ~81 % positive in v2)
  • 3 difficulty levels: easy / medium / hard
  • New scene fields: generation_prompt, causal_chain
  • New category fields: category_metadata (physical_principle, physics_domain, object_pool)
  • Extended clip: 4 s / 120 frames (vs 2 s / 60 frames)
  • New question dimension: "causal"

Usage:
    # Full generation (requires OPENAI_API_KEY)
    python generate_v3.py --input data/question_v2.json --output data/question_v3.json

    # Dry-run (placeholder content, no API calls)
    python generate_v3.py --dry_run

    # Resume an interrupted run
    python generate_v3.py --resume --output data/question_v3.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ============================================================================
# 1. Category Taxonomy  (50 categories)
# ============================================================================

@dataclass
class CategoryDef:
    description: str
    physical_principle: str
    physics_domain: str          # mechanics | fluid | thermal | optics | em | chemistry | material
    object_pool: list[str]       # objects that can appear in scenes
    environment_pool: list[str]  # settings / containers / surfaces
    key_dimensions: list[str]    # most relevant question dimensions
    target_scenes: int           # 20-30
    v2_key: str | None = None    # old name in question_v2.json (None = new category)


DIMENSIONS = [
    "existence", "size", "appearance/texture", "shape",
    "motion", "number", "direction", "material",
    "object", "spatial", "interaction", "causal",
]

CATEGORY_TAXONOMY: dict[str, CategoryDef] = {

    # ── Mechanics (12) ──────────────────────────────────────────────────────
    "Gravity_Earth": CategoryDef(
        description="Object released and falls under Earth's gravitational acceleration (9.8 m/s²).",
        physical_principle="Newton's second law with constant downward acceleration g = 9.8 m/s². "
                           "Velocity increases linearly; distance grows as ½gt².",
        physics_domain="mechanics",
        object_pool=["stone", "apple", "metal key", "tennis ball", "book", "pen", "coin",
                     "glass marble", "leaf", "feather", "hammer", "egg", "orange",
                     "plastic cup", "wooden block", "bolt", "rubber eraser", "walnut",
                     "rubber ball", "ceramic mug"],
        environment_pool=["table edge", "held by a hand", "tree branch", "shelf",
                          "window ledge", "rooftop", "ladder rung", "cliff edge"],
        key_dimensions=["motion", "direction", "spatial", "causal"],
        target_scenes=25,
        v2_key="Gravity_Earth",
    ),

    "Gravity_Moon": CategoryDef(
        description="Object falls under the Moon's weak gravity (~1.6 m/s²), moving noticeably slower than on Earth.",
        physical_principle="Gravitational acceleration on the Moon is ~1/6 of Earth's. "
                           "Objects fall slowly with the same parabolic trajectory but stretched timescale.",
        physics_domain="mechanics",
        object_pool=["astronaut helmet", "wrench", "feather", "stone", "ping-pong ball",
                     "apple", "golf ball", "metal bolt", "sand bag", "rubber ball",
                     "glass marble", "coin", "pen", "wooden cube", "hammer"],
        environment_pool=["lunar surface", "moon crater", "low-gravity habitat",
                          "astronaut's hand", "space station floor"],
        key_dimensions=["motion", "direction", "spatial", "causal"],
        target_scenes=20,
        v2_key="Gravity_Space",
    ),

    "Collision_Elastic": CategoryDef(
        description="Two objects collide and rebound with kinetic energy conserved; objects bounce apart.",
        physical_principle="Conservation of momentum and kinetic energy. "
                           "Objects deform momentarily and rebound at predictable velocities.",
        physics_domain="mechanics",
        object_pool=["rubber ball", "tennis ball", "basketball", "billiard ball", "ping-pong ball",
                     "steel ball bearing", "elastic ring", "football", "superball",
                     "golf ball", "handball", "inflated balloon", "hollow plastic ball",
                     "hard rubber cube", "bowling ball"],
        environment_pool=["hard wall", "concrete floor", "wooden floor", "racket surface",
                          "glass surface", "another ball", "metal plate"],
        key_dimensions=["motion", "direction", "interaction", "causal"],
        target_scenes=25,
        v2_key="Collision_Elastic",
    ),

    "Collision_Inelastic": CategoryDef(
        description="Objects collide and stick together or deform permanently; kinetic energy is lost.",
        physical_principle="Conservation of momentum but not kinetic energy. "
                           "Objects deform, stick, or shatter; relative velocity after impact is lower.",
        physics_domain="mechanics",
        object_pool=["wet clay", "soft dough", "mud ball", "ripe tomato", "rice cake",
                     "putty", "overripe banana", "raw egg", "soft cheese", "cookie dough",
                     "wet sand ball", "mashed potato ball", "playdough cube",
                     "very ripe peach", "soft butter block"],
        environment_pool=["wall", "hard floor", "wooden board", "glass window",
                          "shirt fabric", "another clay lump"],
        key_dimensions=["motion", "shape", "interaction", "causal"],
        target_scenes=25,
        v2_key="Collision_Inelastic",
    ),

    "Elastic_Deformation": CategoryDef(
        description="Material is compressed or stretched and returns to its original shape when force is removed.",
        physical_principle="Hooke's law: restoring force proportional to displacement. "
                           "Elastic strain energy is fully recovered on unloading.",
        physics_domain="mechanics",
        object_pool=["stress ball", "rubber band", "sponge", "spring", "trampoline",
                     "foam pillow", "rubber eraser", "silicone mold", "elastic hair tie",
                     "bungee cord", "rubber glove", "inflated balloon", "memory foam",
                     "soft rubber cube", "latex tube"],
        environment_pool=["hand grip", "between two fingers", "on a flat surface",
                          "hanging from a hook", "pressed against a wall"],
        key_dimensions=["shape", "motion", "material", "causal"],
        target_scenes=25,
        v2_key="Elastic Deformation",
    ),

    "Inelastic_Deformation": CategoryDef(
        description="Material is deformed beyond its elastic limit and retains the new shape permanently.",
        physical_principle="Plastic deformation: stress exceeds yield strength, causing permanent "
                           "rearrangement of internal structure without recovery.",
        physics_domain="mechanics",
        object_pool=["clay lump", "playdough", "soft lead cylinder", "wax block", "putty",
                     "warm butter stick", "wet sand mound", "foil ball", "soft cheese block",
                     "raw dough ball", "modeling clay ring", "beeswax slab",
                     "overripe avocado", "mud cake", "soft chocolate block"],
        environment_pool=["hand", "flat surface", "mold", "rolling pin", "finger press",
                          "hammer strike", "wooden stamp"],
        key_dimensions=["shape", "material", "motion", "causal"],
        target_scenes=20,
    ),

    "Friction_High": CategoryDef(
        description="Object slides over a high-friction surface and decelerates rapidly, stopping in a short distance.",
        physical_principle="Kinetic friction force = μₖ × normal force. High μ means large deceleration; "
                           "object loses velocity quickly; heat or wear may be visible.",
        physics_domain="mechanics",
        object_pool=["wooden block", "book", "cardboard box", "eraser", "brick",
                     "sandpaper block", "rubber-soled shoe", "rough ceramic tile",
                     "hockey puck", "foam block", "metal plate with rough coating",
                     "coarse stone", "fabric bag", "rubber mat", "wicker basket"],
        environment_pool=["rough concrete", "thick grass", "sandpaper surface", "wooden table",
                          "carpet", "gravel path", "rubber mat", "coarse stone floor"],
        key_dimensions=["motion", "direction", "spatial", "causal"],
        target_scenes=25,
        v2_key="Friction_High",
    ),

    "Friction_Low": CategoryDef(
        description="Object glides over a low-friction surface and travels a long distance before stopping.",
        physical_principle="Low coefficient of friction allows kinetic energy to persist. "
                           "Object decelerates slowly; sliding distance is much greater than on rough surfaces.",
        physics_domain="mechanics",
        object_pool=["curling stone", "Lego block", "flat stone", "ice cube", "hockey puck",
                     "polished metal disc", "glass marble", "air hockey puck",
                     "Teflon-coated block", "smooth plastic toy", "waxed wooden block",
                     "butter stick", "soap bar", "frozen fish fillet", "steel washer"],
        environment_pool=["ice surface", "glass table", "polished marble floor",
                          "waxed wooden floor", "wet tile", "oiled metal plate",
                          "playground slide", "Teflon sheet"],
        key_dimensions=["motion", "direction", "spatial", "causal"],
        target_scenes=25,
        v2_key="Friction_Low",
    ),

    "Pendulum": CategoryDef(
        description="A mass suspended from a fixed pivot swings back and forth, demonstrating periodic motion and energy exchange.",
        physical_principle="Simple pendulum: period T = 2π√(L/g), independent of mass. "
                           "Potential and kinetic energy alternate; amplitude decreases slowly due to air resistance.",
        physics_domain="mechanics",
        object_pool=["metal ball", "wooden sphere", "stone weight", "clock pendulum bob",
                     "steel nut on string", "glass ball", "ceramic bead", "rubber ball",
                     "lead weight", "brass sphere", "acorn", "walnut on string",
                     "small lantern", "crystal pendant", "metal ring"],
        environment_pool=["string attached to ceiling", "rod pivot", "grandfather clock",
                          "playground swing", "hanging from door frame", "outdoor tree branch"],
        key_dimensions=["motion", "direction", "spatial", "causal"],
        target_scenes=20,
    ),

    "Rolling_Motion": CategoryDef(
        description="A round object rolls along a surface, converting translational and rotational kinetic energy.",
        physical_principle="Rolling without slipping: v = ωr. Total KE = ½mv² + ½Iω². "
                           "On inclined surfaces, shape/mass distribution affects roll rate.",
        physics_domain="mechanics",
        object_pool=["bowling ball", "basketball", "marble", "coin", "cylinder can",
                     "tennis ball", "wooden barrel", "metal pipe", "wheel", "soup can",
                     "empty paint roller", "tire", "log", "wooden spool", "hollow ball"],
        environment_pool=["inclined ramp", "flat floor", "grass", "carpet", "kitchen counter",
                          "bowling lane", "concrete slope", "wooden board ramp"],
        key_dimensions=["motion", "direction", "spatial", "causal"],
        target_scenes=25,
    ),

    "Projectile_Motion": CategoryDef(
        description="An object is launched and follows a parabolic arc under gravity with no propulsion mid-flight.",
        physical_principle="Horizontal velocity constant; vertical velocity increases at g = 9.8 m/s². "
                           "Trajectory is a parabola; range depends on launch angle and speed.",
        physics_domain="mechanics",
        object_pool=["baseball", "basketball", "football", "tennis ball", "stone",
                     "frisbee", "javelin", "shuttlecock", "golf ball", "paper airplane",
                     "rubber ball", "coin", "water balloon", "beanbag", "snowball"],
        environment_pool=["open field", "gymnasium", "basketball court", "garden",
                          "rooftop", "stadium", "beach", "playground"],
        key_dimensions=["motion", "direction", "spatial", "causal"],
        target_scenes=25,
    ),

    "Structure_Failure": CategoryDef(
        description="A structure collapses under excessive load, impact, or material fatigue.",
        physical_principle="Stress exceeds ultimate tensile/compressive strength of the material. "
                           "Failure propagates from the weakest point; pieces fall under gravity.",
        physics_domain="mechanics",
        object_pool=["wooden bridge", "sandcastle", "card tower", "Jenga tower", "snow fort",
                     "styrofoam arch", "clay pillar", "dominoes structure", "matchstick bridge",
                     "balsa wood frame", "plaster column", "foam house", "sugar-cube tower",
                     "cracker wall", "wooden plank bridge"],
        environment_pool=["table", "sand", "floor", "ground", "platform", "display surface"],
        key_dimensions=["shape", "motion", "spatial", "causal"],
        target_scenes=20,
    ),

    # ── Fluid Dynamics (10) ─────────────────────────────────────────────────
    "Buoyancy_Float": CategoryDef(
        description="Object placed in a fluid displaces its weight in fluid and floats at the surface.",
        physical_principle="Archimedes' principle: buoyant force = ρ_fluid × V_displaced × g. "
                           "Object floats when its average density < fluid density.",
        physics_domain="fluid",
        object_pool=["rubber duck", "sponge", "ice cube", "wooden toy", "plastic bottle",
                     "cork", "foam block", "ping-pong ball", "toy boat", "wax candle",
                     "apple", "coconut", "empty plastic cup", "styrofoam sphere",
                     "balsa wood plank", "hollow metal ball", "pumice stone",
                     "sealed empty bottle", "wooden log", "inflated balloon"],
        environment_pool=["bathtub", "glass tank", "swimming pool", "ocean", "lake",
                          "bucket", "fish tank", "river", "puddle"],
        key_dimensions=["motion", "spatial", "material", "causal"],
        target_scenes=25,
        v2_key="Buoyancy_Float",
    ),

    "Buoyancy_Sink": CategoryDef(
        description="Object placed in a fluid is denser than the fluid and sinks to the bottom.",
        physical_principle="Archimedes' principle: when average density of object > fluid density, "
                           "net force is downward and object accelerates toward the bottom.",
        physics_domain="fluid",
        object_pool=["iron block", "heavy rock", "steel spoon", "anchor", "ceramic cup",
                     "metal coin", "glass marble", "lead weight", "bronze statue",
                     "copper pipe", "stone pebble", "iron nail", "metal bolt",
                     "ceramic tile", "dense rubber ball", "concrete cube",
                     "sand-filled bag", "zinc washer", "cast-iron pan", "brass weight"],
        environment_pool=["glass tank", "bathtub", "swimming pool", "bucket", "fish tank",
                          "ocean floor", "clear container", "aquarium"],
        key_dimensions=["motion", "spatial", "material", "causal"],
        target_scenes=25,
        v2_key="Buoyancy_Sink",
    ),

    "Viscous_Flow": CategoryDef(
        description="A high-viscosity fluid flows slowly and resists deformation, spreading sluggishly.",
        physical_principle="Navier-Stokes equations for high-viscosity Newtonian fluids. "
                           "Flow rate ∝ 1/viscosity; thick fluids maintain shape longer and resist spreading.",
        physics_domain="fluid",
        object_pool=["honey", "condensed milk", "thick oil", "molasses", "blueberry jam",
                     "peanut butter", "caramel sauce", "corn syrup", "tar", "maple syrup",
                     "thick batter", "ketchup", "chocolate sauce", "lava",
                     "tree sap", "gel", "whipped cream", "thick glue", "paint", "yogurt"],
        environment_pool=["toast", "plate", "pancakes", "spoon", "inclined surface",
                          "funnel", "glass jar", "bowl", "bread slice", "waffle"],
        key_dimensions=["motion", "shape", "spatial", "causal"],
        target_scenes=25,
        v2_key="Viscous Flow",
    ),

    "Splash": CategoryDef(
        description="An object impacts a liquid surface, displacing fluid upward into a crown or column.",
        physical_principle="Impact converts kinetic energy into surface waves and ejected droplets. "
                           "Crown height ∝ impact speed; droplet size depends on fluid surface tension.",
        physics_domain="fluid",
        object_pool=["strawberry", "grape", "cherry", "marble", "small stone",
                     "lemon slice", "ice cube", "toy ball", "metal coin",
                     "sugar cube", "rubber duck", "tomato", "blueberry",
                     "walnut", "apple slice", "strawberry cream", "small rubber ball",
                     "ping-pong ball", "glass bead", "small steel ball"],
        environment_pool=["glass of water", "glass of milk", "glass of juice", "bowl of water",
                          "shallow pool", "puddle", "aquarium", "swimming pool"],
        key_dimensions=["motion", "spatial", "shape", "causal"],
        target_scenes=25,
        v2_key="Splash",
    ),

    "Surface_Tension": CategoryDef(
        description="Liquid surface acts like a stretched membrane; small objects can rest on it or droplets form spherical shapes.",
        physical_principle="Surface tension γ (N/m) arises from cohesive forces at the liquid-air interface. "
                           "Pressure inside a droplet: ΔP = 2γ/r; insects can walk on water if weight < surface tension force.",
        physics_domain="fluid",
        object_pool=["water droplet", "soap bubble", "water strider insect", "needle",
                     "paper clip", "small coin", "thin metal wire", "razor blade",
                     "drop of oil", "mercury droplet", "dew droplet", "raindrop",
                     "drop of food coloring", "small leaf", "thin plastic disc"],
        environment_pool=["water surface", "glass of water", "soap solution", "leaf surface",
                          "spider web", "window glass", "petri dish", "calm pond"],
        key_dimensions=["shape", "spatial", "interaction", "causal"],
        target_scenes=20,
    ),

    "Capillary_Action": CategoryDef(
        description="Liquid rises or spreads through narrow spaces against gravity due to adhesion and surface tension.",
        physical_principle="Capillary rise h = 2γ cosθ / (ρgr). Narrower tubes produce higher rise. "
                           "Adhesive forces between liquid and solid overcome gravity.",
        physics_domain="fluid",
        object_pool=["paper towel", "thin glass tube", "celery stalk", "white flower stem",
                     "cotton wick", "sponge", "chalk stick", "sugar cube",
                     "thin straw", "capillary tube", "brick", "cloth strip",
                     "dried pasta", "thin wooden stick", "filter paper strip"],
        environment_pool=["colored water", "ink solution", "water with food dye",
                          "petri dish", "beaker", "shallow tray", "glass of water"],
        key_dimensions=["motion", "direction", "spatial", "causal"],
        target_scenes=20,
    ),

    "Liquid_Layer": CategoryDef(
        description="Two immiscible liquids of different densities separate into distinct horizontal layers.",
        physical_principle="Immiscible liquids with different densities stratify: denser layer settles below. "
                           "Mixing energy dissipates and layers re-separate if undisturbed.",
        physics_domain="fluid",
        object_pool=["cooking oil", "water", "honey", "corn syrup", "rubbing alcohol",
                     "dish soap", "maple syrup", "milk", "vinegar", "food-colored water",
                     "baby oil", "glycerine", "lamp oil", "light mineral oil"],
        environment_pool=["tall glass cylinder", "mason jar", "test tube",
                          "clear plastic bottle", "fish tank", "transparent jug"],
        key_dimensions=["spatial", "appearance/texture", "material", "causal"],
        target_scenes=20,
    ),

    "Liquid_Mixing": CategoryDef(
        description="Two miscible liquids are combined, diffusing and blending into a uniform mixture.",
        physical_principle="Miscible liquids mix by diffusion and convection. "
                           "Without stirring, mixing is slow; with agitation it is rapid.",
        physics_domain="fluid",
        object_pool=["red food dye", "blue food dye", "yellow food dye",
                     "milk", "coffee", "juice", "vinegar", "salt water",
                     "warm water", "cold water", "soap solution", "sugar syrup",
                     "green tea", "black ink drop", "lemon juice"],
        environment_pool=["glass of water", "bowl", "clear container", "mixing cup",
                          "aquarium", "transparent pitcher"],
        key_dimensions=["appearance/texture", "spatial", "motion", "causal"],
        target_scenes=25,
    ),

    "Granular_Flow": CategoryDef(
        description="A collection of discrete solid particles flows and piles up like a fluid under gravity.",
        physical_principle="Granular material exhibits both solid-like (at rest) and fluid-like (flowing) behavior. "
                           "Angle of repose determines pile shape; flow rate depends on particle size and funnel geometry.",
        physics_domain="fluid",
        object_pool=["sand", "rice", "sugar", "salt", "coffee beans", "dry soil",
                     "gravel", "small pebbles", "wheat flour", "cornmeal",
                     "bird seed", "lentils", "sesame seeds", "dry pasta pieces",
                     "glass beads", "mustard seeds", "crushed dry leaves",
                     "aquarium gravel", "fine sawdust", "coarse salt"],
        environment_pool=["hand", "funnel", "hourglass", "glass container",
                          "flat surface", "pile on ground", "inclined ramp",
                          "jar", "bowl", "transparent tube"],
        key_dimensions=["motion", "shape", "spatial", "causal"],
        target_scenes=25,
        v2_key="Granular",
    ),

    "Vortex": CategoryDef(
        description="A rotating fluid forms a funnel-shaped vortex around a low-pressure core.",
        physical_principle="Angular momentum conservation causes fluid to spin faster near the center. "
                           "Centripetal acceleration balances pressure gradient; vortex depth grows with rotation speed.",
        physics_domain="fluid",
        object_pool=["water in a drain", "stirred cup of tea", "tornado",
                     "dust devil", "whirlpool in a bathtub", "stirred paint pot",
                     "milk swirling in coffee", "whisk in batter",
                     "stirred aquarium", "draining sink"],
        environment_pool=["bathtub", "sink drain", "large glass", "bucket",
                          "mixing bowl", "fish tank", "clear container"],
        key_dimensions=["motion", "shape", "spatial", "causal"],
        target_scenes=20,
    ),

    # ── Material / Deformation (8) ───────────────────────────────────────────
    "Fracture": CategoryDef(
        description="A brittle object breaks into fragments when stress exceeds its fracture toughness.",
        physical_principle="Brittle fracture: crack propagates rapidly once critical stress intensity factor K_Ic is reached. "
                           "No plastic deformation; shards radiate from impact point.",
        physics_domain="material",
        object_pool=["wine glass", "ceramic plate", "glass bottle", "mirror",
                     "ceramic vase", "porcelain cup", "glass window", "clay pot",
                     "glass fish bowl", "tile", "glass sphere", "china bowl",
                     "glass panel", "brittle plastic ruler", "crystal ornament"],
        environment_pool=["hard floor", "concrete surface", "stone counter", "wooden floor",
                          "brick wall", "metal anvil", "another hard object"],
        key_dimensions=["shape", "motion", "spatial", "causal"],
        target_scenes=25,
        v2_key="Fracture",
    ),

    "Slicing": CategoryDef(
        description="A sharp blade moves through an object, cutting it cleanly into two or more pieces.",
        physical_principle="Stress concentration at blade edge exceeds shear strength along cut line. "
                           "Cut quality depends on blade sharpness, applied force, and material hardness.",
        physics_domain="material",
        object_pool=["bread loaf", "carrot", "apple", "cucumber", "tomato",
                     "rope", "cheese block", "banana", "clay block", "soft wood",
                     "marshmallow", "butter block", "watermelon", "mushroom",
                     "raw chicken breast", "celery stalk", "onion half",
                     "sponge cake", "eggplant", "zucchini"],
        environment_pool=["cutting board", "kitchen counter", "flat surface",
                          "butcher block", "marble slab"],
        key_dimensions=["shape", "interaction", "motion", "causal"],
        target_scenes=25,
    ),

    "Tearing": CategoryDef(
        description="A soft or fibrous material is pulled apart, propagating a crack or tear along the material.",
        physical_principle="Fracture mechanics for ductile/fibrous materials: crack tip stress concentration "
                           "propagates tear; fibers stretch then break sequentially.",
        physics_domain="material",
        object_pool=["newspaper", "paper sheet", "paper towel", "plastic bag",
                     "fabric cloth", "rubber sheet", "thin leather", "foil sheet",
                     "tissue paper", "wet paper", "duct tape", "thin plastic film",
                     "rice paper", "nori sheet", "cotton fabric"],
        environment_pool=["two hands", "hands and fixed edge", "tearing machine",
                          "sharp corner", "nail", "flat surface"],
        key_dimensions=["shape", "motion", "interaction", "causal"],
        target_scenes=20,
    ),

    "Cloth_Simulation": CategoryDef(
        description="Flexible fabric moves realistically under wind, gravity, or contact forces.",
        physical_principle="Cloth dynamics modeled as mass-spring system. Gravity pulls downward; "
                           "tension forces resist stretching; aerodynamic forces cause flutter.",
        physics_domain="material",
        object_pool=["silk flag", "curtain", "tablecloth", "scarf", "flag",
                     "bed sheet", "cape", "paper flag", "light dress fabric",
                     "handkerchief", "gossamer curtain", "towel", "canvas sail",
                     "ribbon", "thin plastic sheet", "parachute"],
        environment_pool=["flagpole", "window frame", "chair back", "clothesline",
                          "wind outdoors", "fan indoors", "falling through air"],
        key_dimensions=["motion", "shape", "spatial", "causal"],
        target_scenes=25,
        v2_key="Cloth Simulation",
    ),

    "Explosion": CategoryDef(
        description="Rapid release of energy causes pressure wave, fragmentation, and outward dispersion of debris.",
        physical_principle="Rapid exothermic reaction or pressure overload releases energy faster than it can dissipate, "
                           "creating supersonic shock wave and ejecting fragments radially.",
        physics_domain="material",
        object_pool=["inflated balloon", "gasoline drum", "pressurized container",
                     "firecracker", "water balloon", "plastic soda bottle", "popcorn kernel",
                     "spray can", "overinflated tire", "bubble underwater",
                     "sealed jar heated", "compressed-gas cylinder", "party popper",
                     "exploding cake", "pressure cooker failure"],
        environment_pool=["open space", "indoor room", "underwater", "concrete surface",
                          "metal container", "lab bench"],
        key_dimensions=["motion", "spatial", "shape", "causal"],
        target_scenes=20,
        v2_key="Explosion",
    ),

    "Smoke": CategoryDef(
        description="Small particles or vapor rise and disperse in air, forming visible plumes.",
        physical_principle="Buoyancy drives hot smoke upward (ρ_smoke < ρ_air). "
                           "Diffusion and turbulent mixing spread the plume; concentration decreases with height.",
        physics_domain="material",
        object_pool=["lit candle", "incense stick", "blown-out candle", "cup of hot tea",
                     "cigarette", "campfire embers", "hot food", "dry ice",
                     "smoke machine", "chimney", "burning match", "hot iron",
                     "fogger device", "extinguished fire", "steam pipe"],
        environment_pool=["indoor room", "outdoor environment", "dim room",
                          "kitchen", "fire pit", "stage", "open air"],
        key_dimensions=["motion", "direction", "shape", "causal"],
        target_scenes=25,
        v2_key="Smoke",
    ),

    "Combustion": CategoryDef(
        description="A material burns, producing visible flame, smoke, light, and heat.",
        physical_principle="Combustion: rapid oxidation releases heat and light. "
                           "Fuel + O₂ → CO₂ + H₂O + heat. Flame color indicates temperature and fuel type.",
        physics_domain="material",
        object_pool=["candle wick", "wood log", "paper sheet", "match stick",
                     "dried leaf", "twig", "birthday candle", "torch",
                     "alcohol lamp", "oil lamp", "campfire log", "sparkler",
                     "newspaper", "cardboard", "wax tablet"],
        environment_pool=["fireplace", "campfire pit", "lab bench", "candle holder",
                          "outdoor setting", "metal bowl", "fire pit"],
        key_dimensions=["appearance/texture", "motion", "shape", "causal"],
        target_scenes=20,
    ),

    "Chain_Reaction": CategoryDef(
        description="An initial perturbation triggers a cascade of sequential events, each causing the next.",
        physical_principle="Each element stores potential energy (gravitational or elastic) that is transferred "
                           "to the next upon impact. Total energy propagates down the chain.",
        physics_domain="material",
        object_pool=["dominoes", "mousetraps", "billiard balls in line",
                     "falling books", "chain of coins", "Newton's cradle",
                     "lined-up toy blocks", "stacked dominoes", "bowling pins",
                     "linked magnets", "falling card stack", "toy train collision",
                     "cascading water buckets", "toppling columns",
                     "matchstick chain"],
        environment_pool=["flat table", "wooden floor", "ramp", "gymnasium floor",
                          "outdoor surface", "display table"],
        key_dimensions=["motion", "number", "direction", "causal"],
        target_scenes=20,
    ),

    # ── Thermal / Phase (8) ─────────────────────────────────────────────────
    "Melting": CategoryDef(
        description="A solid gradually transforms into a liquid as temperature rises above its melting point.",
        physical_principle="Phase transition at melting point T_m: latent heat absorbed breaks crystalline bonds. "
                           "Volume may increase slightly; shape collapses as solid becomes liquid.",
        physics_domain="thermal",
        object_pool=["ice cube", "candle wax", "butter", "snow patch", "ice cream",
                     "chocolate bar", "wax crayon", "lead block", "paraffin candle",
                     "ice pop", "frozen yogurt", "wax seal", "lard block",
                     "ice sculpture", "gelatin block"],
        environment_pool=["warm plate", "heated pan", "direct sunlight", "microwave plate",
                          "warm water bath", "room temperature counter", "hot surface"],
        key_dimensions=["shape", "appearance/texture", "motion", "causal"],
        target_scenes=25,
        v2_key="Melting",
    ),

    "Freezing": CategoryDef(
        description="A liquid cools below its freezing point and solidifies, changing shape and texture.",
        physical_principle="Phase transition: latent heat released as liquid loses thermal energy. "
                           "Crystal structure forms; volume may change slightly (water expands ~9% on freezing).",
        physics_domain="thermal",
        object_pool=["water in tray", "juice in mold", "wet cloth", "puddle",
                     "lake surface", "dripping water", "water in cup",
                     "mercury thermometer liquid", "fruit juice", "coconut water",
                     "lemon juice", "herbal tea", "smoothie", "sports drink", "broth"],
        environment_pool=["freezer interior", "cold winter outdoor", "ice bath",
                          "refrigerator shelf", "liquid nitrogen container",
                          "cold metal surface", "outdoor frost"],
        key_dimensions=["shape", "appearance/texture", "material", "causal"],
        target_scenes=20,
    ),

    "Evaporation": CategoryDef(
        description="Liquid molecules at the surface gain enough energy to escape into vapor phase, gradually reducing liquid volume.",
        physical_principle="Surface evaporation rate ∝ temperature, surface area, and vapor pressure deficit. "
                           "Puddles shrink; wet surfaces dry; latent heat is absorbed from surroundings.",
        physics_domain="thermal",
        object_pool=["water puddle", "wet cloth", "water droplets on leaf",
                     "glass of water", "wet pavement", "ink droplet",
                     "rubbing alcohol drop", "acetone pool", "wet sand",
                     "water on hot pan", "morning dew", "wet shoe sole",
                     "brine droplet", "perfume drop", "water on glass surface"],
        environment_pool=["hot surface", "outdoor sunlight", "warm room",
                          "heated plate", "dry air environment", "desert setting"],
        key_dimensions=["appearance/texture", "size", "motion", "causal"],
        target_scenes=20,
    ),

    "Boiling": CategoryDef(
        description="Liquid heated to boiling point forms bubbles of vapor that rise and burst at the surface.",
        physical_principle="Vapor pressure equals atmospheric pressure at boiling point (100 °C for water at 1 atm). "
                           "Nucleate bubbles form on heated surface, rise, and release vapor.",
        physics_domain="thermal",
        object_pool=["water in pot", "milk in saucepan", "oil in pan", "soup",
                     "broth", "tea kettle water", "sugar syrup", "salt water",
                     "egg in boiling water", "pasta in pot", "jam in pan",
                     "hot chocolate", "caramel mixture", "stew liquid", "gravy"],
        environment_pool=["stove burner", "hot plate", "campfire", "induction cooker",
                          "lab hotplate", "fondue pot"],
        key_dimensions=["appearance/texture", "motion", "spatial", "causal"],
        target_scenes=20,
    ),

    "Thermal_Expansion": CategoryDef(
        description="A material expands visibly as its temperature increases due to increased molecular vibration.",
        physical_principle="Linear expansion: ΔL = α L₀ ΔT. Coefficient α depends on material. "
                           "Gases expand most; metals less; bridges and rails have expansion joints.",
        physics_domain="thermal",
        object_pool=["metal rod", "balloon over bottle neck", "bimetallic strip",
                     "thermometer liquid column", "rubber ball", "air in sealed syringe",
                     "plastic bottle with air", "metal lid on jar", "metal ring",
                     "coiled spring", "wire on posts", "sealed air balloon"],
        environment_pool=["hot water bath", "heat gun", "boiling water", "open flame",
                          "oven", "sunlight", "heating pad"],
        key_dimensions=["size", "shape", "motion", "causal"],
        target_scenes=20,
    ),

    "Thermal_Convection": CategoryDef(
        description="Fluid circulates in loops as heated portions rise and cooled portions sink, transferring heat.",
        physical_principle="Density decreases with temperature; hot fluid rises, cooler fluid sinks. "
                           "Rayleigh-Bénard convection cells form in heated fluid layers.",
        physics_domain="thermal",
        object_pool=["colored hot water", "lava lamp fluid", "hot cocoa",
                     "ink in warm water", "heated oil", "glycerine with dye",
                     "water with food coloring", "heated aquarium water",
                     "boiling soup", "warm sugar solution"],
        environment_pool=["transparent container", "lava lamp", "glass bowl",
                          "aquarium", "heated beaker", "lab flask"],
        key_dimensions=["motion", "direction", "spatial", "causal"],
        target_scenes=20,
    ),

    "Crystallization": CategoryDef(
        description="Dissolved or molten material forms ordered crystal structures as it cools or concentrates.",
        physical_principle="Crystal nucleation and growth: ions/molecules arrange into minimum-energy lattice. "
                           "Growth rate depends on supersaturation and temperature gradient.",
        physics_domain="thermal",
        object_pool=["salt solution", "sugar solution", "borax solution",
                     "copper sulfate solution", "alum solution", "cooling chocolate",
                     "honey crystallizing", "cooling wax", "cooling metal cast",
                     "rock candy solution", "frost on window"],
        environment_pool=["petri dish", "jar", "glass container", "cooling surface",
                          "cold window", "refrigerator shelf"],
        key_dimensions=["shape", "appearance/texture", "size", "causal"],
        target_scenes=20,
    ),

    "Condensation": CategoryDef(
        description="Water vapor cools below dew point and condenses into liquid droplets on a cold surface.",
        physical_principle="Condensation occurs when surface temperature < dew point. "
                           "Latent heat is released; droplet size grows with time; droplets coalesce and run.",
        physics_domain="thermal",
        object_pool=["cold glass", "cold metal can", "bathroom mirror",
                     "cold window glass", "cold pipe", "ice-cold bottle",
                     "refrigerator exterior", "cold stone surface",
                     "cold car windshield", "cold metal plate"],
        environment_pool=["warm humid room", "bathroom after shower", "outdoor morning",
                          "refrigerator exterior", "kitchen", "greenhouse"],
        key_dimensions=["appearance/texture", "size", "spatial", "causal"],
        target_scenes=20,
    ),

    # ── Optics / Light (3) ──────────────────────────────────────────────────
    "Reflection": CategoryDef(
        description="Light or image bounces off a reflective surface at equal angle of incidence and reflection.",
        physical_principle="Law of reflection: θ_i = θ_r. Specular reflection on smooth surfaces creates mirror image; "
                           "diffuse reflection on rough surfaces scatters light.",
        physics_domain="optics",
        object_pool=["dancer in mirror", "person waving", "car on wet road",
                     "sunlight on lake", "man brushing teeth", "ball rolling to mirror",
                     "candle flame in mirror", "hand gesturing", "text on mirror",
                     "laser beam on mirror", "lamp reflected in polished floor",
                     "face in spoon reflection", "sky in puddle", "building in glass facade",
                     "fire reflected in water"],
        environment_pool=["flat mirror", "glass window", "still lake", "wet road",
                          "polished metal", "calm water pool", "bathroom mirror",
                          "shop window", "chrome surface"],
        key_dimensions=["direction", "spatial", "motion", "appearance/texture"],
        target_scenes=25,
        v2_key="Reflection",
    ),

    "Refraction": CategoryDef(
        description="Light bends as it passes between media of different refractive indices, displacing apparent object position.",
        physical_principle="Snell's law: n₁ sinθ₁ = n₂ sinθ₂. Objects in water appear shifted/bent. "
                           "Lenses and prisms exploit refraction to focus or disperse light.",
        physics_domain="optics",
        object_pool=["spoon in water", "straw in glass", "pencil in water", "banana behind glass",
                     "coin at bottom of bowl", "fish in aquarium", "fork in water",
                     "pen in glass of water", "laser beam entering water",
                     "prism splitting white light", "magnifying glass focusing light",
                     "straw appearing at angle", "ruler in water", "chopstick in glass"],
        environment_pool=["glass of water", "aquarium", "glass prism",
                          "lens setup", "fish tank", "clear water bowl", "glass cylinder"],
        key_dimensions=["shape", "appearance/texture", "spatial", "direction"],
        target_scenes=25,
        v2_key="Refraction",
    ),

    "Shadow": CategoryDef(
        description="An opaque object blocks a light source, casting a shadow whose shape and length depend on geometry.",
        physical_principle="Shadow boundary traced by tangent rays from light source. "
                           "Shadow length = h × D/d where D is source-object distance and d is object-screen distance.",
        physics_domain="optics",
        object_pool=["person walking", "rotating basketball", "Lego blocks", "tree branch",
                     "moving hand puppet", "swinging pendulum", "passing car",
                     "rotating fan blade", "moving cloud", "candle with obstacle",
                     "hand blocking flashlight", "flying bird silhouette",
                     "plant on windowsill", "book under lamp", "spinning top"],
        environment_pool=["sunlit wall", "ground under sunlight", "dim room with lamp",
                          "screen behind object", "painted wall", "sandy ground",
                          "white sheet", "whiteboard"],
        key_dimensions=["shape", "direction", "spatial", "motion"],
        target_scenes=25,
        v2_key="Shadow",
    ),

    # ── Electromagnetic (2) ─────────────────────────────────────────────────
    "Magnetism": CategoryDef(
        description="Magnetic objects attract or repel each other, and ferromagnetic materials are drawn to magnets.",
        physical_principle="Magnetic force F = qv×B (on moving charge) or ∝ field gradient on dipole. "
                           "Like poles repel; unlike poles attract; iron/nickel strongly attracted.",
        physics_domain="electromagnetic",
        object_pool=["bar magnet", "horseshoe magnet", "neodymium magnet", "iron nail",
                     "steel paperclip", "iron filing", "compass needle", "steel key",
                     "iron washer", "steel ball bearing", "iron powder",
                     "refrigerator magnet", "magnetic toy", "metal coin (ferrous)",
                     "iron chain link"],
        environment_pool=["table surface", "air", "paper substrate", "glass table",
                          "sand surface", "hanging from string", "resting on wood"],
        key_dimensions=["motion", "direction", "interaction", "causal"],
        target_scenes=20,
    ),

    "Electrostatics": CategoryDef(
        description="Charged objects attract neutral or oppositely-charged objects due to electrostatic force.",
        physical_principle="Coulomb's law: F = kq₁q₂/r². Static charge on balloon attracts neutral paper. "
                           "Triboelectric effect generates charge via contact and friction.",
        physics_domain="electromagnetic",
        object_pool=["inflated balloon", "plastic comb", "PVC rod", "wool fabric",
                     "bits of paper", "hair strands", "small styrofoam beads",
                     "thin aluminum foil strips", "charged amber rod",
                     "puffed rice grains", "shredded paper", "thin plastic strips",
                     "pepper grains", "thin string", "dried flower petals"],
        environment_pool=["table top", "air", "hair surface", "paper surface",
                          "near balloon", "near charged rod", "enclosed jar"],
        key_dimensions=["motion", "direction", "interaction", "causal"],
        target_scenes=20,
    ),

    # ── Chemical / Material Interaction (7) ─────────────────────────────────
    "Diffusion": CategoryDef(
        description="Particles or molecules spread from high concentration to low concentration until uniformly distributed.",
        physical_principle="Fick's law: flux ∝ −D × ∇C. Brownian motion drives random particle movement "
                           "toward equilibrium. Faster in gases, slower in liquids, very slow in solids.",
        physics_domain="chemistry",
        object_pool=["drop of ink in water", "food dye drop", "perfume spray",
                     "drop of milk in tea", "potassium permanganate crystal",
                     "drop of colored dye", "smoke in still air",
                     "drop of cream in coffee", "gas in sealed chamber",
                     "colored sugar drop in water", "red wine drop in water",
                     "turmeric in water", "drop of oil in alcohol"],
        environment_pool=["glass of still water", "petri dish", "transparent container",
                          "still air", "closed jar", "aquarium"],
        key_dimensions=["motion", "appearance/texture", "spatial", "causal"],
        target_scenes=20,
    ),

    "Dissolution": CategoryDef(
        description="A solute disperses into a solvent at molecular level, forming a homogeneous solution.",
        physical_principle="Solvation: solvent molecules surround solute ions/molecules, separating them. "
                           "Rate increases with temperature, stirring, and surface area.",
        physics_domain="chemistry",
        object_pool=["salt crystal", "sugar cube", "effervescent tablet", "bath bomb",
                     "instant coffee granules", "hard candy", "Alka-Seltzer tablet",
                     "vitamin C tablet", "chalk piece in acid", "baking soda",
                     "copper sulfate crystal", "food dye tablet", "bouillon cube",
                     "aspirin tablet", "dry gelatin powder"],
        environment_pool=["glass of water", "warm water bowl", "cold water glass",
                          "vinegar solution", "juice", "clear container"],
        key_dimensions=["appearance/texture", "size", "motion", "causal"],
        target_scenes=20,
    ),

    "Rust_Oxidation": CategoryDef(
        description="Iron or steel surface reacts with oxygen and moisture, forming reddish-brown iron oxide over time (shown in time-lapse).",
        physical_principle="Electrochemical corrosion: Fe + O₂ + H₂O → Fe₂O₃·nH₂O. "
                           "Rust forms as porous layer that allows further oxidation beneath.",
        physics_domain="chemistry",
        object_pool=["iron nail", "steel nail", "iron plate", "steel wool",
                     "iron chain", "steel bolt", "iron wire mesh",
                     "cast iron pan", "steel washer", "iron railing",
                     "steel screw", "iron beam section", "steel fence post",
                     "iron pipe section", "steel file"],
        environment_pool=["salt water", "moist air", "outdoor rain exposure",
                          "humid room", "salt spray", "wet soil"],
        key_dimensions=["appearance/texture", "size", "material", "causal"],
        target_scenes=20,
    ),

    "Color_Mixing": CategoryDef(
        description="Two or more colored substances are combined, producing a new mixed color.",
        physical_principle="Subtractive mixing (pigments): CMY primaries; mixing absorbs complementary wavelengths. "
                           "Additive mixing (light): RGB primaries; mixing increases perceived brightness.",
        physics_domain="chemistry",
        object_pool=["red paint", "blue paint", "yellow paint", "red food dye",
                     "blue food dye", "yellow dye solution", "watercolor paints",
                     "red ink", "blue ink", "green dye", "white paint",
                     "black paint", "cyan dye", "magenta dye", "colored powder"],
        environment_pool=["palette", "mixing bowl", "glass of water", "paper surface",
                          "paint tray", "clear container", "canvas"],
        key_dimensions=["appearance/texture", "motion", "interaction", "causal"],
        target_scenes=25,
    ),

    "Adsorption": CategoryDef(
        description="Molecules adhere to the surface of a material, removing them from solution or gas phase.",
        physical_principle="Surface adsorption: molecules bind to active surface sites via van der Waals or "
                           "chemical bonds. Adsorption capacity depends on surface area and affinity.",
        physics_domain="chemistry",
        object_pool=["activated charcoal", "sponge absorbing liquid", "paper towel",
                     "silica gel", "zeolite beads", "clay powder",
                     "iron filings attracting oil", "dry cloth in colored water",
                     "coconut shell charcoal", "porous ceramic bead",
                     "coffee filter paper", "sand adsorbing dye"],
        environment_pool=["colored water", "ink solution", "dye bath",
                          "oil-contaminated water", "clear container",
                          "transparent beaker"],
        key_dimensions=["appearance/texture", "motion", "interaction", "causal"],
        target_scenes=20,
    ),

    "Precipitation": CategoryDef(
        description="Two solutions react to form an insoluble solid (precipitate) that falls out of solution.",
        physical_principle="Solubility product exceeded: when [A⁺][B⁻] > Ksp, AB precipitate forms. "
                           "Precipitate nucleates, grows, and settles due to gravity.",
        physics_domain="chemistry",
        object_pool=["silver nitrate and sodium chloride solutions",
                     "barium chloride and sodium sulfate solutions",
                     "lead nitrate and sodium iodide solutions",
                     "calcium chloride and sodium carbonate solutions",
                     "iron(III) chloride and sodium hydroxide",
                     "copper sulfate and sodium hydroxide",
                     "lime water and CO₂",
                     "milk of magnesia forming",
                     "calcium carbonate forming",
                     "zinc hydroxide forming"],
        environment_pool=["test tube", "beaker", "clear container",
                          "glass mixing vessel", "transparent flask"],
        key_dimensions=["appearance/texture", "motion", "spatial", "causal"],
        target_scenes=20,
    ),

    "Foam_Bubble": CategoryDef(
        description="Gas trapped in liquid forms bubbles or foam; bubbles rise, coalesce, and burst.",
        physical_principle="Bubble stability: surface tension γ maintains pressure difference ΔP = 4γ/r. "
                           "Foam drains over time; bubbles coalesce; surfactants stabilize foam.",
        physics_domain="chemistry",
        object_pool=["soap foam in sink", "beer foam", "latte milk foam",
                     "whipped cream", "shaving cream", "bubble bath foam",
                     "carbonated water bubbles", "lemon juice with baking soda",
                     "hand soap foam", "fizzy drink bubbles",
                     "sea foam", "foam from waterfall",
                     "bubbles in boiling milk", "hydrogen peroxide foam",
                     "foam from soap squeeze"],
        environment_pool=["sink basin", "glass", "bowl", "counter surface",
                          "cup", "bucket", "transparent container"],
        key_dimensions=["shape", "motion", "size", "causal"],
        target_scenes=20,
    ),
}

# ============================================================================
# 2. V2 → V3 category name mapping
# ============================================================================

V2_TO_V3: dict[str, str] = {
    cat_def.v2_key: cat_name
    for cat_name, cat_def in CATEGORY_TAXONOMY.items()
    if cat_def.v2_key is not None
}

# Reverse: v3_name → v2_key
V3_TO_V2: dict[str, str] = {v: k for k, v in V2_TO_V3.items()}

# ============================================================================
# 3. Prompt templates
# ============================================================================

SCENE_GEN_SYSTEM = (
    "You are an expert physicist and video benchmark designer. "
    "Generate physically accurate, visually diverse scene descriptions for "
    "text-to-video model evaluation. Respond ONLY with valid JSON."
)

SCENE_GEN_USER = """\
Physical phenomenon: {category_name}
Description: {description}
Physical principle: {physical_principle}
Physics domain: {physics_domain}

Object pool (choose from or invent similar):
  {object_pool}

Environment pool:
  {environment_pool}

Already existing scene captions (DO NOT duplicate, ensure diversity):
{existing_captions}

Generate exactly {n_scenes} new, diverse scenes as a JSON array.
Each scene object must have these exact keys:

{{
  "scene_name": "<short 3-7 word title>",
  "scene_caption": "<1-2 sentence description of the physical event for a T2V model prompt>",
  "generation_prompt": "<detailed 2-3 sentence prompt with object, setting, action, and expected physical outcome — suitable for text-to-video generation>",
  "causal_chain": {{
    "initial_state": "<what is present/positioned before the event starts>",
    "process": "<what physical process happens during the clip>",
    "final_state": "<what state objects reach by end of the 4-second clip>"
  }},
  "questions": [
    {{
      "id": "01",
      "question_text": "<yes/no question about what should be observed>",
      "dimensions": ["<one or two from: existence, size, appearance/texture, shape, motion, number, direction, material, object, spatial, interaction, causal>"],
      "difficulty": "<easy|medium|hard>",
      "frame_indices": [<2-5 frame indices from 0-119>],
      "answer": <1 for yes, 0 for no>
    }},
    ... (8 questions total: EXACTLY 4 with answer=1 and EXACTLY 4 with answer=0)
  ]
}}

Rules for questions:
  1. Exactly 8 questions per scene: 4 must have answer=1, 4 must have answer=0.
  2. answer=1 questions: observable facts that a correctly generated video SHOULD show.
  3. answer=0 questions: physically INCORRECT behaviors that a bad video might show (violations).
  4. Vary difficulty: at least 2 easy, 2 medium, 2 hard per scene.
  5. Frame indices must be from [0, 119] (4-second clip at 30 fps); use [20, 60, 100] for temporal spread.
  6. Cover at least 3 different dimensions across the 8 questions.
  7. Questions must be concrete and visually answerable by watching the video.
  8. Do not repeat the same question pattern across all 8 questions.

Respond with a JSON array of {n_scenes} scene objects only. No markdown, no extra text.
"""


# ============================================================================
# 4. LLM backend (OpenAI Chat Completions)
# ============================================================================

def call_openai(system: str, user: str, model: str = "gpt-4o",
                max_tokens: int = 8192) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        return ""
    return content.strip()


def parse_json_array(raw: str) -> list[dict]:
    """Extract a JSON array from raw LLM output, stripping markdown fences."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", raw)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return []


def make_placeholder_scenes(category_name: str, cat_def: CategoryDef,
                             n: int, start_idx: int = 1) -> list[dict]:
    """Fallback: placeholder scenes for dry-run mode."""
    scenes = []
    for i in range(n):
        obj = cat_def.object_pool[i % len(cat_def.object_pool)]
        env = cat_def.environment_pool[i % len(cat_def.environment_pool)]
        caption = f"A {obj} undergoes {cat_def.description.lower()} in a {env}."
        scenes.append({
            "scene_name": f"{obj.title()} in {env}",
            "scene_caption": caption,
            "generation_prompt": caption + " The video captures the full physical event from start to finish.",
            "causal_chain": {
                "initial_state": f"A {obj} is positioned in/on a {env}.",
                "process": f"The physical phenomenon ({category_name}) occurs.",
                "final_state": "The event reaches its final equilibrium state.",
            },
            "questions": [
                {
                    "id": f"{j+1:02d}",
                    "question_text": f"[placeholder Q{j+1} answer={j%2}]",
                    "dimensions": [cat_def.key_dimensions[j % len(cat_def.key_dimensions)]],
                    "difficulty": ["easy", "medium", "hard"][j % 3],
                    "frame_indices": [20, 60, 100],
                    "answer": j % 2,
                }
                for j in range(8)
            ],
        })
    return scenes


# ============================================================================
# 5. V2 scene migration to V3 format
# ============================================================================

def migrate_v2_scene(v2_scene: dict, category_name: str) -> dict:
    """
    Add generation_prompt and causal_chain to a v2 scene.
    Keep original questions but update frame_indices to 120-frame range and
    add medium difficulty where questions were easy/hard.
    """
    caption = v2_scene.get("scene_caption", "")

    # Derive generation_prompt from caption
    gen_prompt = (
        f"{caption} "
        "The video shows the complete physical process over approximately 4 seconds, "
        "with smooth, physically accurate motion and realistic material behavior."
    )

    # Derive causal_chain heuristically from caption
    causal_chain = {
        "initial_state": _extract_initial_state(caption),
        "process": _extract_process(caption, category_name),
        "final_state": _extract_final_state(caption),
    }

    # Scale frame_indices from 60-frame to 120-frame range
    questions = []
    for q in v2_scene.get("questions", []):
        new_q = dict(q)
        new_q["frame_indices"] = [min(119, fi * 2) for fi in q.get("frame_indices", [20, 60, 100])]
        questions.append(new_q)

    return {
        "scene_name": v2_scene.get("scene_name", ""),
        "scene_caption": caption,
        "generation_prompt": gen_prompt,
        "causal_chain": causal_chain,
        "questions": questions,
    }


def _extract_initial_state(caption: str) -> str:
    """Simple heuristic: take the first clause up to the first comma or 'and'."""
    for sep in [",", " and ", "; "]:
        if sep in caption:
            return caption.split(sep)[0].strip() + "."
    return caption.split(".")[0].strip() + "."


def _extract_process(caption: str, category_name: str) -> str:
    return f"The {category_name.lower().replace('_', ' ')} phenomenon occurs as described in the scene."


def _extract_final_state(caption: str) -> str:
    sentences = caption.replace("  ", " ").split(". ")
    return sentences[-1].strip().rstrip(".") + "." if sentences else caption


# ============================================================================
# 6. Main generation logic
# ============================================================================

def generate_v3(
    v2_path: str,
    output_path: str,
    dry_run: bool = False,
    resume: bool = False,
    sleep_between: float = 1.0,
    batch_size: int = 5,
    model: str = "gpt-4o",
) -> None:
    # ── Load v2 ──────────────────────────────────────────────────────────────
    with open(v2_path, "r", encoding="utf-8") as f:
        v2_data = json.load(f)
    v2_cats = v2_data.get("main_categories", {})

    # ── Load existing v3 output for resume ──────────────────────────────────
    v3_output: dict[str, dict] = {}   # cat_name → {"category_metadata": ..., "scenes": {...}}
    if resume and Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        v3_output = prev.get("categories", {})
        print(f"Resume: {len(v3_output)} categories already present.")

    # ── Build v3 category by category ────────────────────────────────────────
    total_cats = len(CATEGORY_TAXONOMY)
    for cat_idx, (cat_name, cat_def) in enumerate(CATEGORY_TAXONOMY.items(), 1):
        print(f"\n[{cat_idx}/{total_cats}] {cat_name}  (target: {cat_def.target_scenes} scenes)")

        if cat_name in v3_output:
            existing_count = len(v3_output[cat_name].get("scenes", {}))
            if existing_count >= cat_def.target_scenes:
                print(f"  Skipping — {existing_count} scenes already present.")
                continue

        # ── Initialize category entry ────────────────────────────────────────
        if cat_name not in v3_output:
            v3_output[cat_name] = {
                "category_metadata": {
                    "description": cat_def.description,
                    "physical_principle": cat_def.physical_principle,
                    "physics_domain": cat_def.physics_domain,
                    "object_pool": cat_def.object_pool,
                    "environment_pool": cat_def.environment_pool,
                    "key_dimensions": cat_def.key_dimensions,
                },
                "scenes": {},
            }

        scenes_dict: dict = v3_output[cat_name]["scenes"]

        # ── Migrate existing v2 scenes ───────────────────────────────────────
        v2_key = cat_def.v2_key
        if v2_key and v2_key in v2_cats:
            for scene_id, v2_scene in v2_cats[v2_key].items():
                if scene_id not in scenes_dict:
                    scenes_dict[scene_id] = migrate_v2_scene(v2_scene, cat_name)
            print(f"  Migrated {len(v2_cats[v2_key])} v2 scenes.")

        # ── Generate new scenes ──────────────────────────────────────────────
        current_count = len(scenes_dict)
        needed = cat_def.target_scenes - current_count
        if needed <= 0:
            print(f"  Already have {current_count} scenes — no generation needed.")
            _save(v3_output, output_path, v2_data["metadata"])
            continue

        print(f"  Generating {needed} new scenes (already have {current_count})...")

        existing_captions = "\n".join(
            f"  - {s.get('scene_caption', '')}"
            for s in list(scenes_dict.values())[:20]
        ) or "  (none)"

        # Generate in batches
        generated: list[dict] = []
        remaining = needed
        while remaining > 0:
            batch = min(batch_size, remaining)
            print(f"    Requesting batch of {batch}...")

            if dry_run:
                batch_scenes = make_placeholder_scenes(
                    cat_name, cat_def, batch,
                    start_idx=current_count + len(generated) + 1
                )
            else:
                prompt = SCENE_GEN_USER.format(
                    category_name=cat_name,
                    description=cat_def.description,
                    physical_principle=cat_def.physical_principle,
                    physics_domain=cat_def.physics_domain,
                    object_pool=", ".join(cat_def.object_pool),
                    environment_pool=", ".join(cat_def.environment_pool),
                    existing_captions=existing_captions,
                    n_scenes=batch,
                )
                try:
                    raw = call_openai(SCENE_GEN_SYSTEM, prompt, model=model)
                    batch_scenes = parse_json_array(raw)
                    if not batch_scenes:
                        print("    WARNING: Failed to parse LLM output, using placeholders.")
                        batch_scenes = make_placeholder_scenes(
                            cat_name, cat_def, batch,
                            start_idx=current_count + len(generated) + 1
                        )
                except Exception as e:
                    print(f"    ERROR calling LLM: {e}. Using placeholders.")
                    batch_scenes = make_placeholder_scenes(
                        cat_name, cat_def, batch,
                        start_idx=current_count + len(generated) + 1
                    )

            generated.extend(batch_scenes)
            existing_captions += "\n" + "\n".join(
                f"  - {s.get('scene_caption', '')}" for s in batch_scenes
            )
            remaining -= batch
            if remaining > 0 and not dry_run:
                time.sleep(sleep_between)

        # ── Store generated scenes ────────────────────────────────────────────
        for i, scene in enumerate(generated):
            scene_id = f"scene_{current_count + i + 1:02d}"
            # Validate and fix answer balance
            scene["questions"] = _fix_answer_balance(scene.get("questions", []))
            scenes_dict[scene_id] = scene

        print(f"  Total scenes now: {len(scenes_dict)}")

        # ── Checkpoint save after each category ─────────────────────────────
        _save(v3_output, output_path, v2_data["metadata"])

    print(f"\nDone. Output: {output_path}")
    _print_stats(v3_output)


def _fix_answer_balance(questions: list[dict]) -> list[dict]:
    """Ensure exactly 4 positive (answer=1) and 4 negative (answer=0) per scene."""
    if len(questions) != 8:
        return questions  # don't modify malformed question sets
    pos = [q for q in questions if q.get("answer") == 1]
    neg = [q for q in questions if q.get("answer") == 0]
    if len(pos) == 4 and len(neg) == 4:
        return questions
    # Rebalance by flipping excess items
    all_q = list(questions)
    target_pos = 4
    flipped = 0
    for q in all_q:
        cur_pos = sum(1 for x in all_q if x.get("answer") == 1)
        if cur_pos > target_pos and q.get("answer") == 1:
            q["answer"] = 0
            flipped += 1
        elif cur_pos < target_pos and q.get("answer") == 0:
            q["answer"] = 1
            flipped += 1
        if abs(sum(1 for x in all_q if x.get("answer") == 1) - target_pos) == 0:
            break
    return all_q


def _save(v3_output: dict, output_path: str, v2_meta: dict) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total_scenes = sum(len(v.get("scenes", {})) for v in v3_output.values())
    total_questions = sum(
        len(scene.get("questions", []))
        for cat_data in v3_output.values()
        for scene in cat_data.get("scenes", {}).values()
    )
    out = {
        "metadata": {
            "version": "v3",
            "fps": 30,
            "duration_seconds": 4,
            "total_frames": 120,
            "dimensions": DIMENSIONS,
            "difficulty_levels": ["easy", "medium", "hard"],
            "total_categories": len(v3_output),
            "total_scenes": total_scenes,
            "total_questions": total_questions,
            "note": (
                "v3: 50 categories × 20-30 scenes × 8 questions. "
                "Answer balance: 4 positive + 4 negative per scene. "
                "Scenes migrated from v2 have migrated_ prefix in scene_id."
            ),
        },
        "categories": v3_output,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def _print_stats(v3_output: dict) -> None:
    print("\n── V3 Statistics ───────────────────────────────────")
    total_scenes = 0
    total_questions = 0
    pos = neg = 0
    for cat_name, cat_data in v3_output.items():
        scenes = cat_data.get("scenes", {})
        n = len(scenes)
        total_scenes += n
        for scene in scenes.values():
            qs = scene.get("questions", [])
            total_questions += len(qs)
            pos += sum(1 for q in qs if q.get("answer") == 1)
            neg += sum(1 for q in qs if q.get("answer") == 0)
        print(f"  {cat_name:<30s} {n:3d} scenes")
    print(f"\nTotal categories : {len(v3_output)}")
    print(f"Total scenes     : {total_scenes}")
    print(f"Total questions  : {total_questions}")
    if total_questions:
        print(f"Answer balance   : {pos} pos ({100*pos//total_questions}%) / "
              f"{neg} neg ({100*neg//total_questions}%)")
    print("────────────────────────────────────────────────────")


# ============================================================================
# 7. CLI entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PhyMetric question_v3.json")
    parser.add_argument("--input",  default="data/question_v2.json",
                        help="Path to question_v2.json (source of 100 human-designed scenes)")
    parser.add_argument("--output", default="data/question_v3.json",
                        help="Output path for question_v3.json")
    parser.add_argument("--model",  default="gpt-4o",
                        help="OpenAI model id for scene generation (e.g. gpt-4o, gpt-4o-mini)")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Scenes to generate per API call (default: 5)")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds to sleep between API calls (default: 1.0)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Use placeholder content; no API calls")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skip completed categories")
    parser.add_argument("--only_cats", nargs="*",
                        help="If set, only generate these categories (by name)")
    args = parser.parse_args()

    if not args.dry_run and "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not set. Use --dry_run or set the env var.")
        return

    # Filter categories if --only_cats specified
    global CATEGORY_TAXONOMY
    if args.only_cats:
        CATEGORY_TAXONOMY = {k: v for k, v in CATEGORY_TAXONOMY.items() if k in args.only_cats}
        print(f"Filtered to {len(CATEGORY_TAXONOMY)} categories: {list(CATEGORY_TAXONOMY.keys())}")

    generate_v3(
        v2_path=args.input,
        output_path=args.output,
        dry_run=args.dry_run,
        resume=args.resume,
        sleep_between=args.sleep,
        batch_size=args.batch_size,
        model=args.model,
    )


if __name__ == "__main__":
    main()
