Frame 0    Frame 1    Frame 2    ...    Frame T
   ↓          ↓          ↓                 ↓
[ViT]      [ViT]      [ViT]            [ViT]
   ↓          ↓          ↓                 ↓
   └──────────┴──────────┴────────────────┘
                      ↓
              Temporal Features (B, T, N, D)
                      ↓
    ┌─────────────────────────────────────────────┐
    │  TEXT-CONDITIONED QUERIES                    │
    │                                              │
    │  "interacted object" ──→ [Query] ──→ boxes  │
    │  "robot gripper"     ──→ [Query] ──→ boxes  │
    │  "robot arm"         ──→ [Query] ──→ boxes  │
    │  "start location"    ──→ [Query] ──→ box    │
    │  "end location"      ──→ [Query] ──→ box    │
    │  "target container"  ──→ [Query] ──→ box    │
    │                                              │
    └─────────────────────────────────────────────┘
                      ↓
              Boxes + Features per query
                      ↓
                 [LLM Reasoning]
                      ↓
    "The [red cube]<box> was picked by [gripper]<box>..."