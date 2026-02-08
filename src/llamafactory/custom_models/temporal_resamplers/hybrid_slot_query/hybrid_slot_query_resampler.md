Video → Vision Encoder → [Slot Attention] → discovered objects
                              ↓
                    [Typed Queries conditioned on slots]
                              ↓
                    start_loc, end_loc, target, robot
                              ↓
                         [LLM Reasoning]
                              ↓
                    Grounded description with boxes



┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: DISCOVERY (Slot Attention - discovers WHAT moved)        │
│                                                                     │
│  • Slots compete for visual features                               │
│  • Motion-biased: slots naturally gravitate to moving things       │
│  • Output: 1-2 "active" slots with temporal tracks                 │
│  • Temporal consistency via SlotContrast-style loss                │
│                                                                     │
│  Output: discovered_slots (B, num_discovered, H)                   │
│          slot_boxes (B, num_discovered, T, 4)                      │
│          slot_masks (B, T, num_discovered, H, W)                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    Discovered slots as CONDITIONING
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: RELATIONS (Q-Former style queries conditioned on slots)  │
│                                                                     │
│  Fixed typed queries:                                               │
│  • [START_LOC] - where did the object come from?                   │
│  • [END_LOC] - where did the object go?                            │
│  • [TARGET] - what is the target/goal location?                    │
│  • [ROBOT] - where is the robot/gripper?                           │
│                                                                     │
│  These queries cross-attend to:                                     │
│  • Visual features (to find locations)                             │
│  • Discovered slots (to relate to moved object)                    │
│                                                                     │
│  Output: relation_features (B, num_queries, H)                     │
│          relation_boxes (B, num_queries, 4)                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: LLM REASONING                                            │
│                                                                     │
│  Input tokens: [slot_0] [slot_1] [start] [end] [target] [robot]   │
│  + text prompt                                                      │
│                                                                     │
│  Output: "The [red cube]<box> was picked from [table]<box>         │
│           and placed in [blue bowl]<box>"                          │
└─────────────────────────────────────────────────────────────────────┘




                    Losses:
Loss	What it does	Supervision needed
Slot Box Loss	Match best slot to GT moved object	Moved object boxes (B, T, 4)
Query Box Loss	Each query predicts correct location	Start/end/target boxes
Slot Classification	Classify slot as robot/object/background	Slot labels (optional)
Temporal Consistency	Same slot = same object across frames	None (self-supervised)
Language Loss	Generate correct description	Text annotations