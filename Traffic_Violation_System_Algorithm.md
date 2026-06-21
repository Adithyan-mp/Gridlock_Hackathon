# Intelligent Traffic Violation Analysis and Evidence Generation System
## Algorithmic Description (Proof of Concept)

---

## Algorithm 1: Image Acquisition

**Input:** Camera feed `C`
**Output:** Image `I`, Metadata `M`

```
1. Capture frame I from camera feed C
2. Record metadata M = {camera_id, timestamp, GPS_location}
3. Return I, M
```

---

## Algorithm 2: Adaptive Image Preprocessing

**Input:** Raw image `I`
**Output:** Preprocessed image `I'`

```
1. Compute blur_score   ← Laplacian_Variance(I)
2. Compute brightness   ← Mean_Intensity(I)
3. Compute noise_level  ← Noise_Estimation(I)
4. Compute contrast     ← Contrast_Assessment(I)

5. If contrast = LOW:
       I ← CLAHE(I)
6. If noise_level = HIGH:
       I ← Denoise(I)
7. If brightness = LOW:
       I ← Gamma_Correction(I)

8. I ← Resize(I, target_size)
9. I ← Convert_RGB(I)
10. I' ← Normalize(I)

11. Return I'
```
**Design Rationale:** Steps 5–7 are conditional, not unconditional, to prevent loss of fine detail on already-clean frames, which would otherwise degrade detector accuracy.

---

## Algorithm 3: Vehicle and Road-User Detection

**Input:** Preprocessed image `I'`
**Output:** Detection set `D = {d₁, d₂, ..., dₙ}`

```
1. classes ← {Car, Bus, Truck, Motorcycle, Bicycle,
               Auto-rickshaw, Person, Traffic_Light, License_Plate}
2. D ← YOLO_Detect(I', classes)
3. For each detection d in D:
       d.label, d.confidence, d.bbox ← model output
4. Return D
```

---

## Algorithm 4: Road Infrastructure Detection

**Input:** Preprocessed image `I'`
**Output:** Infrastructure set `R`

```
1. lanes      ← Detect_Lane_Markings(I')
2. stop_lines ← Detect_Stop_Lines(I')
3. crossings  ← Detect_Pedestrian_Crossings(I')
4. zones      ← Detect_Parking_Zones(I')
5. signs      ← Detect_Traffic_Signs(I')
6. R ← {lanes, stop_lines, crossings, zones, signs}
7. Return R
```

---

## Algorithm 5: Multi-Object Tracking

**Input:** Detection sequence `D₁, D₂, ..., Dₜ` (per frame)
**Output:** Tracked object set `T` with persistent IDs

```
1. Initialize tracker ← ByteTrack() | DeepSORT()
2. For t = 1 to current_frame:
       T ← tracker.Update(Dₜ)
       For each object o in T:
           Append o.bbox to o.trajectory_history
3. Return T
```

---

## Algorithm 6: Vehicle–Rider Association

**Input:** Vehicle `v`, Candidate riders `P = {p₁, ..., pₖ}`
**Output:** Associated rider `r` or NULL

```
1. best_score ← -∞ ; best_rider ← NULL
2. For each p in P:
       IoU_score      ← IoU(v.bbox, p.bbox)
       dist_score     ← 1 / Center_Distance(v, p)
       motion_score   ← Motion_Consistency(v.track, p.track)
       history_score  ← Tracking_History_Overlap(v.id, p.id)

       score(p) ← w₁·IoU_score + w₂·dist_score +
                   w₃·motion_score + w₄·history_score

       If score(p) > best_score:
           best_score ← score(p) ; best_rider ← p

3. If best_score > θ_association:
       Return best_rider
   Else:
       Return NULL
```

---

## Algorithm 7: Violation Detection (7 Types)

**Input:** Tracked objects `T`, Infrastructure `R`, Signal state `S`
**Output:** Candidate violation set `V`

```
1. V ← {}
2. For each vehicle v in T.vehicles:

   // A. Helmet
   3. If v.type = Motorcycle:
          r ← Vehicle_Rider_Association(v, T.persons)        [Algorithm 6]
          If r ≠ NULL and Helmet_Detected(r) = FALSE:
              V ← V ∪ {Violation(HELMET, v, r)}

   // B. Seatbelt
   4. If v.type ∈ {Car, Truck, Bus}:
          region ← Extract_Driver_Region(v)
          If Seatbelt_Detected(region) = FALSE:
              V ← V ∪ {Violation(SEATBELT, v)}

   // C. Triple Riding
   5. If v.type = Motorcycle:
          count ← Count_Associated_Riders(v, T.persons)
          If count > 2:
              V ← V ∪ {Violation(TRIPLE_RIDING, v)}

   // D. Wrong-Side Driving
   6. lane_dir    ← Lane_Direction(R.lanes, v.position)
      vehicle_dir ← Motion_Vector(v.trajectory_history)
      If Angle(lane_dir, vehicle_dir) > 90°:
          V ← V ∪ {Violation(WRONG_SIDE, v)}

   // E. Stop-Line Violation
   7. bc ← Bottom_Center(v.bbox)
      If S = RED and Crosses(bc, R.stop_lines):
          V ← V ∪ {Violation(STOP_LINE, v)}

   // F. Red-Light Violation
   8. If S = RED:
          Begin_Monitoring(v)
      If v crosses R.stop_lines while monitored and S = RED:
          V ← V ∪ {Violation(RED_LIGHT, v)}

   // G. Illegal Parking
   9. If Inside(v.position, R.zones[NO_PARKING]):
          duration ← Stationary_Duration(v.trajectory_history)
          If duration > θ_parking:
              V ← V ∪ {Violation(ILLEGAL_PARKING, v)}

3. Return V
```
**Note (Step 6):** Direction comparison, not box overlap, avoids false positives from vehicles merely adjacent to a lane.
**Note (Step 7):** Bottom-center point used instead of full bounding-box overlap, for the same reason.
**Note (Step 8):** No fixed delay (e.g., 15s) after red activation — monitoring begins immediately.

---

## Algorithm 8: Violation Reasoning Engine (Key Innovation)

**Input:** Candidate violations `V`, Tracked objects `T`, Infrastructure `R`
**Output:** Confirmed violations `V'` with severity and confidence

```
1. V' ← {}
2. For each candidate v in V:

   3. Build scene graph G:
          Nodes(G) ← {v.vehicle, v.rider, lane, stop_line,
                       traffic_signal, parking_zone}
          Edges(G) ← Spatial_Relations(G) ∪ Temporal_Relations(G)

   4. type       ← Infer_Violation_Type(G)
   5. severity   ← Infer_Severity(G)
   6. confidence ← Infer_Confidence(G)

   7. V' ← V' ∪ {(type, severity, confidence, G)}

3. Return V'
```
**Design Rationale:** Reasoning is performed jointly over relationships between all entities in the scene graph, rather than evaluating each rule in isolation — producing context-aware, defensible violation calls.

---

## Algorithm 9: License Plate Recognition

**Input:** Image `I`, Violating vehicle `v` (triggered only if `v ∈ V'`)
**Output:** Registration number `N`

```
1. plate_bbox ← YOLO_Detect_Plate(I, region = v.bbox)
2. crop ← Crop(I, plate_bbox)
3. crop ← Resize(crop)
4. crop ← Grayscale(crop)
5. crop ← Contrast_Enhance(crop)
6. crop ← Denoise(crop)
7. crop ← Threshold(crop)
8. crop ← Perspective_Correct(crop)
9. crop ← Sharpen(crop)
10. N ← OCR_Engine(crop)        // PaddleOCR / EasyOCR
11. Return N
```
**Design Rationale:** Triggered only on confirmed violations — avoids unnecessary OCR cost on the majority of compliant traffic.

---

## Algorithm 10: Vehicle Information Retrieval

**Input:** Registration number `N`
**Output:** Vehicle record `Info`

```
1. Info ← Database_Lookup(N)
2. Return {vehicle_type: Info.type,
           registration_details: Info.details,
           owner_name: Info.owner}
```
**Limitation:** Owner identity ≠ driver identity; driver cannot be determined from a plate alone.

---

## Algorithm 11: Duplicate Violation Suppression

**Input:** New violation record `v_new`, Violation log `L`
**Output:** Boolean `is_duplicate`

```
1. For each v_past in L:
       If v_past.vehicle_number = v_new.vehicle_number AND
          v_past.violation_type = v_new.violation_type AND
          v_past.location = v_new.location AND
          |v_past.timestamp − v_new.timestamp| < Δt_window:
              Return TRUE
2. Return FALSE
```
**Note:** Different violation types from the same vehicle/location are still recorded independently.

---

## Algorithm 12: Evidence Generation

**Input:** Image `I`, Violation `v`, Plate info, Metadata `M`
**Output:** Annotated evidence image `E`

```
1. E ← Draw_BBox(I, v.vehicle.bbox, "Vehicle")
2. E ← Draw_BBox(E, v.rider.bbox, "Rider/Driver")
3. E ← Draw_BBox(E, plate.bbox, "Plate")
4. E ← Overlay_Text(E, v.type, v.confidence,
                     M.timestamp, M.camera_id, M.location)
5. Return E
```

---

## Algorithm 13: Metadata Generation

**Input:** Violation `v`, Vehicle info, Registration `N`, Metadata `M`, Evidence path `P`
**Output:** Structured record `Rec`

```
1. Rec ← {
       vehicle_number:   N,
       owner_name:       vehicle_info.owner_name,
       vehicle_type:     vehicle_info.vehicle_type,
       violation_type:   v.type,
       confidence_score: v.confidence,
       timestamp:        M.timestamp,
       camera_id:        M.camera_id,
       evidence_path:    P
   }
2. Return Rec   // structured JSON
```

---

## Algorithm 14: Analytics Dashboard Update

**Input:** Violation record `Rec`, Dashboard database `DB`
**Output:** Updated analytics view

```
1. DB.Insert(Rec)
2. daily        ← DB.Aggregate_By_Day()
3. trends       ← DB.Compute_Trends()
4. high_risk    ← DB.Rank_Locations_By_Count()
5. repeat_off   ← DB.Find_Repeat_Offenders()
6. report       ← DB.Generate_Summary_Report()
7. Return {daily, trends, high_risk, repeat_off, report}
```

---

## Algorithm 15: Performance Evaluation

**Input:** Predictions `P`, Ground truth `G`, Runtime logs `Log`
**Output:** Metric report `M`

```
1. Detection   ← {Precision(P,G), Recall(P,G), F1(P,G), mAP(P,G)}
2. Tracking    ← {MOTA(P,G), IDF1(P,G)}
3. OCR         ← {Char_Accuracy(P,G), Plate_Accuracy(P,G)}
4. System      ← {Log.latency, Log.fps, Log.memory, Log.scalability}
5. Return M ← {Detection, Tracking, OCR, System}
```

---

## Master Algorithm: End-to-End Pipeline Orchestration

**Input:** Camera feed `C`, Violation log `L`, Dashboard `DB`
**Output:** Updated violation log `L`

```
1. (I, M) ← Image_Acquisition(C)                              [Algorithm 1]
2. I' ← Adaptive_Preprocessing(I)                              [Algorithm 2]
3. D ← Detect_Vehicles_Road_Users(I')                          [Algorithm 3]
4. R ← Detect_Road_Infrastructure(I')                          [Algorithm 4]
5. T ← Multi_Object_Tracking(D)                                [Algorithm 5]
6. For each vehicle v in T.vehicles:
       v.rider ← Vehicle_Rider_Association(v, T.persons)       [Algorithm 6]

7. S ← Read_Traffic_Light_State(D)
8. V ← Violation_Detection(T, R, S)                            [Algorithm 7]
9. If V = ∅: Return L                       // no further processing

10. V' ← Reasoning_Engine(V, T, R)                             [Algorithm 8]

11. For each violation v in V':
        N      ← License_Plate_Recognition(I, v.vehicle)        [Algorithm 9]
        Info   ← Vehicle_Info_Retrieval(N)                      [Algorithm 10]
        v_new  ← Build_Record(v, N, M)

        If Duplicate_Check(v_new, L) = TRUE:                    [Algorithm 11]
            Continue to next violation

        E      ← Evidence_Generation(I, v, plate, M)            [Algorithm 12]
        P      ← Save(E)
        Rec    ← Metadata_Generation(v, Info, N, M, P)          [Algorithm 13]

        L ← L ∪ {Rec}
        DB ← Dashboard_Update(Rec, DB)                          [Algorithm 14]

12. Return L
```

*Algorithm 15 (Performance Evaluation) runs as a scheduled offline job over accumulated predictions and logs, not per-frame.*

---

## Complexity Summary

| Stage | Dominant Cost | Approx. Complexity |
|-------|---------------|---------------------|
| Preprocessing | Per-image filtering | O(W×H) |
| Detection (YOLO) | Forward pass | O(1) per frame (fixed-size CNN) |
| Tracking | Association across frames | O(n²) per frame (n = objects) |
| Rider Association | Pairwise scoring | O(v×p) (v=vehicles, p=persons) |
| Violation Detection | Rule evaluation per vehicle | O(n) |
| Reasoning Engine | Graph construction + inference | O(n²) (graph edges) |
| OCR | Triggered only on violations | O(k), k ≪ n |

---

## POC Validity Conditions

For this system to be considered functionally complete as a proof of concept, the following must hold:
1. Detection (Algorithm 3) achieves acceptable mAP on the 9 target classes.
2. Tracking (Algorithm 5) maintains ID consistency across occlusion (IDF1 above threshold).
3. Reasoning Engine (Algorithm 8) produces confidence scores correlated with manual review agreement.
4. Duplicate Suppression (Algorithm 11) reduces redundant records without suppressing genuinely distinct violations.
5. End-to-end latency (Algorithm 15) supports near-real-time operation across multiple camera streams.
