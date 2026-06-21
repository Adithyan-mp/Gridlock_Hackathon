"""
INTELLIGENT TRAFFIC VIOLATION ANALYSIS AND EVIDENCE GENERATION SYSTEM
Pseudocode — End-to-End Pipeline (15 Stages)
"""

# ============================================================
# STAGE 1: IMAGE ACQUISITION
# ============================================================

FUNCTION acquire_image(camera_feed):
    image = camera_feed.capture()
    metadata = {
        camera_id: camera_feed.id,
        timestamp: current_time(),
        location: camera_feed.gps_location
    }
    RETURN image, metadata


# ============================================================
# STAGE 2: ADAPTIVE IMAGE PREPROCESSING
# ============================================================

FUNCTION preprocess_image(image):

    # --- Quality assessment ---
    blur_score      = compute_laplacian_variance(image)
    brightness      = compute_brightness(image)
    noise_level     = estimate_noise(image)
    contrast_level  = assess_contrast(image)

    # --- Conditional correction (only fix what is broken) ---
    IF contrast_level == LOW:
        image = apply_CLAHE(image)

    IF noise_level == HIGH:
        image = apply_denoising(image)

    IF brightness == LOW:
        image = apply_gamma_correction(image)

    # --- Standard normalization (always applied) ---
    image = resize(image, model_input_size)
    image = convert_to_RGB(image)
    image = normalize_pixels(image)

    RETURN image

    # NOTE: Enhancements are applied conditionally, not blanket,
    # to avoid degrading detector accuracy on already-clean frames.


# ============================================================
# STAGE 3: VEHICLE & ROAD-USER DETECTION
# ============================================================

FUNCTION detect_vehicles_and_road_users(image):
    CLASSES = [Car, Bus, Truck, Motorcycle, Bicycle,
               AutoRickshaw, Person, TrafficLight, LicensePlate]

    detections = YOLO_model.predict(image, classes=CLASSES)
    # Each detection -> {class_label, confidence_score, bbox}
    RETURN detections


# ============================================================
# STAGE 4: ROAD INFRASTRUCTURE DETECTION
# ============================================================

FUNCTION detect_road_infrastructure(image):
    lane_markings        = detect_lanes(image)
    stop_lines           = detect_stop_lines(image)
    pedestrian_crossings = detect_crossings(image)
    parking_zones        = detect_parking_zones(image)
    traffic_signs        = detect_signs(image)

    RETURN {
        lanes: lane_markings,
        stop_lines: stop_lines,
        crossings: pedestrian_crossings,
        parking_zones: parking_zones,
        signs: traffic_signs
    }


# ============================================================
# STAGE 5: MULTI-OBJECT TRACKING
# ============================================================

FUNCTION track_objects(detections_per_frame):
    tracker = ByteTrack()   # or DeepSORT()

    FOR EACH frame_detections IN detections_per_frame:
        tracked_objects = tracker.update(frame_detections)
        # Assigns persistent track_id, maintains trajectory history

    RETURN tracked_objects   # list of {track_id, class, bbox_history}


# ============================================================
# STAGE 6: VEHICLE–RIDER ASSOCIATION
# ============================================================

FUNCTION associate_rider_to_vehicle(vehicle, candidate_riders):
    best_match = NULL
    best_score = -INFINITY

    FOR EACH rider IN candidate_riders:
        iou_score        = compute_IoU(vehicle.bbox, rider.bbox)
        distance_score   = inverse(center_point_distance(vehicle, rider))
        motion_score     = motion_consistency(vehicle.track, rider.track)
        history_score    = tracking_history_overlap(vehicle.track_id, rider.track_id)

        association_score = weighted_sum(
            iou_score, distance_score, motion_score, history_score
        )

        IF association_score > best_score:
            best_score = association_score
            best_match = rider

    IF best_score > ASSOCIATION_THRESHOLD:
        RETURN best_match
    ELSE:
        RETURN NULL


# ============================================================
# STAGE 7: VIOLATION DETECTION (7 TYPES)
# ============================================================

FUNCTION detect_violations(tracked_objects, infrastructure, traffic_light_state):
    violations = []

    FOR EACH vehicle IN tracked_objects.vehicles:

        # --- A. Helmet Violation ---
        IF vehicle.type == Motorcycle:
            rider = associate_rider_to_vehicle(vehicle, tracked_objects.persons)
            IF rider EXISTS AND NOT detect_helmet(rider):
                violations.append(Violation(HELMET, vehicle, rider))

        # --- B. Seatbelt Violation ---
        IF vehicle.type IN [Car, Truck, Bus]:
            driver_region = extract_driver_region(vehicle)
            IF NOT detect_seatbelt(driver_region):
                violations.append(Violation(SEATBELT, vehicle))

        # --- C. Triple Riding ---
        IF vehicle.type == Motorcycle:
            rider_count = count_associated_riders(vehicle, tracked_objects.persons)
            IF rider_count > 2:
                violations.append(Violation(TRIPLE_RIDING, vehicle))

        # --- D. Wrong-Side Driving ---
        lane_direction   = estimate_lane_direction(infrastructure.lanes, vehicle.position)
        vehicle_direction = get_motion_vector(vehicle.track_history)
        IF direction_opposes(vehicle_direction, lane_direction):
            violations.append(Violation(WRONG_SIDE, vehicle))

        # --- E. Stop-Line Violation ---
        bottom_center = get_bottom_center_point(vehicle.bbox)
        IF traffic_light_state == RED AND crosses_line(bottom_center, infrastructure.stop_lines):
            violations.append(Violation(STOP_LINE, vehicle))
            # NOTE: bottom-center point used instead of full bbox overlap
            # to avoid false positives

        # --- F. Red-Light Violation ---
        IF traffic_light_state == RED:
            start_monitoring(vehicle)
            IF crosses_line_after_red(vehicle, infrastructure.stop_lines):
                violations.append(Violation(RED_LIGHT, vehicle))
                # NOTE: no arbitrary fixed delay used; monitoring begins
                # immediately when signal turns red

        # --- G. Illegal Parking ---
        IF is_inside_zone(vehicle.position, infrastructure.parking_zones, type=NO_PARKING):
            stationary_duration = get_stationary_duration(vehicle.track_history)
            IF stationary_duration > PARKING_THRESHOLD:
                violations.append(Violation(ILLEGAL_PARKING, vehicle))

    RETURN violations


# ============================================================
# STAGE 8: VIOLATION REASONING ENGINE (KEY INNOVATION)
# ============================================================

FUNCTION reasoning_engine(violations, tracked_objects, infrastructure):
    refined_violations = []

    FOR EACH v IN violations:

        # --- Build scene graph for this candidate violation ---
        scene_graph = build_graph(
            nodes = [v.vehicle, v.rider, infrastructure.lanes,
                     infrastructure.stop_lines, traffic_signal_node,
                     infrastructure.parking_zones],
            edges = spatial_and_temporal_relationships(
                        v.vehicle, infrastructure, tracked_objects)
        )

        # --- Joint reasoning over the graph ---
        violation_type  = infer_violation_type(scene_graph)
        severity        = infer_severity(scene_graph)
        confidence      = infer_confidence(scene_graph)

        refined_violations.append({
            type: violation_type,
            severity: severity,
            confidence: confidence,
            evidence_links: scene_graph
        })

    RETURN refined_violations
    # Goes beyond flat rule-triggers: reasons jointly over relationships
    # between every entity in the scene, not isolated bounding boxes.


# ============================================================
# STAGE 9: LICENSE PLATE RECOGNITION
# ============================================================

FUNCTION recognize_license_plate(image, vehicle):
    # Triggered ONLY when a violation is confirmed

    # Stage 9.1 — Detect plate
    plate_bbox = YOLO_model.detect_plate(image, region=vehicle.bbox)

    # Stage 9.2 — Preprocess
    plate_crop = crop(image, plate_bbox)
    plate_crop = resize(plate_crop)
    plate_crop = grayscale(plate_crop)
    plate_crop = enhance_contrast(plate_crop)
    plate_crop = denoise(plate_crop)
    plate_crop = threshold(plate_crop)
    plate_crop = perspective_correct(plate_crop)
    plate_crop = sharpen(plate_crop)

    # Stage 9.3 — OCR
    registration_number = OCR_engine.extract_text(plate_crop)  # PaddleOCR / EasyOCR

    RETURN registration_number


# ============================================================
# STAGE 10: VEHICLE INFORMATION RETRIEVAL
# ============================================================

FUNCTION retrieve_vehicle_info(registration_number):
    record = vehicle_database.lookup(registration_number)

    RETURN {
        vehicle_type: record.vehicle_type,
        registration_details: record.registration_details,
        owner_name: record.owner_name
        # NOTE: owner != driver; driver identity cannot be
        # determined from a license plate alone
    }


# ============================================================
# STAGE 11: DUPLICATE VIOLATION SUPPRESSION
# ============================================================

FUNCTION is_duplicate(new_violation, violation_log):
    FOR EACH past_violation IN violation_log:
        IF (past_violation.vehicle_number == new_violation.vehicle_number AND
            past_violation.violation_type == new_violation.violation_type AND
            past_violation.location == new_violation.location AND
            within_time_window(past_violation.timestamp, new_violation.timestamp)):
            RETURN TRUE

    RETURN FALSE
    # Different violation types from the same vehicle are still
    # recorded independently.


# ============================================================
# STAGE 12: EVIDENCE GENERATION
# ============================================================

FUNCTION generate_evidence_image(image, violation, plate_info, metadata):
    annotated_image = draw_bbox(image, violation.vehicle.bbox, label="Vehicle")
    annotated_image = draw_bbox(annotated_image, violation.rider.bbox, label="Rider/Driver")
    annotated_image = draw_bbox(annotated_image, plate_info.bbox, label="Plate")
    annotated_image = overlay_text(annotated_image,
        violation_type = violation.type,
        confidence = violation.confidence,
        timestamp = metadata.timestamp,
        camera_id = metadata.camera_id,
        location = metadata.location
    )
    RETURN annotated_image


# ============================================================
# STAGE 13: METADATA GENERATION
# ============================================================

FUNCTION generate_metadata_record(violation, vehicle_info, plate_number,
                                   metadata, evidence_path):
    record = {
        vehicle_number: plate_number,
        owner_name: vehicle_info.owner_name,
        vehicle_type: vehicle_info.vehicle_type,
        violation_type: violation.type,
        confidence_score: violation.confidence,
        timestamp: metadata.timestamp,
        camera_id: metadata.camera_id,
        evidence_path: evidence_path
    }
    RETURN record   # structured JSON


# ============================================================
# STAGE 14: ANALYTICS DASHBOARD
# ============================================================

FUNCTION update_dashboard(violation_record, dashboard_db):
    dashboard_db.insert(violation_record)

    daily_violations   = dashboard_db.aggregate_by_day()
    violation_trends   = dashboard_db.compute_trends()
    high_risk_locations = dashboard_db.rank_locations_by_violation_count()
    repeat_offenders   = dashboard_db.find_repeat_offenders()

    RETURN {
        daily_violations: daily_violations,
        trends: violation_trends,
        high_risk_locations: high_risk_locations,
        repeat_offenders: repeat_offenders,
        searchable_records: dashboard_db.all_records(),
        summary_report: generate_summary_report(dashboard_db)
    }


# ============================================================
# STAGE 15: PERFORMANCE EVALUATION
# ============================================================

FUNCTION evaluate_system(predictions, ground_truth, runtime_logs):

    detection_metrics = {
        precision: compute_precision(predictions, ground_truth),
        recall:    compute_recall(predictions, ground_truth),
        f1_score:  compute_f1(predictions, ground_truth),
        mAP:       compute_mAP(predictions, ground_truth)
    }

    tracking_metrics = {
        MOTA: compute_MOTA(predictions, ground_truth),
        IDF1: compute_IDF1(predictions, ground_truth)
    }

    ocr_metrics = {
        character_accuracy: compute_char_accuracy(predictions, ground_truth),
        plate_accuracy:     compute_plate_accuracy(predictions, ground_truth)
    }

    system_metrics = {
        latency:      runtime_logs.avg_latency,
        fps:          runtime_logs.avg_fps,
        memory_usage: runtime_logs.peak_memory,
        scalability:  runtime_logs.multi_camera_throughput
    }

    RETURN {
        detection: detection_metrics,
        tracking: tracking_metrics,
        ocr: ocr_metrics,
        system: system_metrics
    }


# ============================================================
# MAIN PIPELINE ORCHESTRATION
# ============================================================

FUNCTION run_pipeline(camera_feed, violation_log, dashboard_db):

    # Stage 1
    image, metadata = acquire_image(camera_feed)

    # Stage 2
    image = preprocess_image(image)

    # Stage 3 & 4
    detections     = detect_vehicles_and_road_users(image)
    infrastructure = detect_road_infrastructure(image)

    # Stage 5
    tracked_objects = track_objects(detections)

    # Stage 6
    FOR EACH vehicle IN tracked_objects.vehicles:
        vehicle.rider = associate_rider_to_vehicle(vehicle, tracked_objects.persons)

    # Stage 7
    traffic_light_state = read_traffic_light_state(detections)
    candidate_violations = detect_violations(tracked_objects, infrastructure, traffic_light_state)

    IF candidate_violations IS EMPTY:
        RETURN  # no further processing needed for compliant traffic

    # Stage 8
    confirmed_violations = reasoning_engine(candidate_violations, tracked_objects, infrastructure)

    FOR EACH violation IN confirmed_violations:

        # Stage 9
        plate_number = recognize_license_plate(image, violation.vehicle)

        # Stage 10
        vehicle_info = retrieve_vehicle_info(plate_number)

        # Stage 11
        candidate_record = build_candidate_record(violation, plate_number, metadata)
        IF is_duplicate(candidate_record, violation_log):
            CONTINUE   # skip to next violation

        # Stage 12
        evidence_image = generate_evidence_image(image, violation, plate_info, metadata)
        evidence_path   = save_image(evidence_image)

        # Stage 13
        metadata_record = generate_metadata_record(
            violation, vehicle_info, plate_number, metadata, evidence_path
        )
        violation_log.append(metadata_record)

        # Stage 14
        update_dashboard(metadata_record, dashboard_db)

    RETURN violation_log


# ============================================================
# STAGE 15 — RUN PERIODICALLY (NOT PER-FRAME)
# ============================================================

FUNCTION scheduled_evaluation_job():
    metrics = evaluate_system(get_recent_predictions(), get_ground_truth(), get_runtime_logs())
    log_metrics(metrics)
