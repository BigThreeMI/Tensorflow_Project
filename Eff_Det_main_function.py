def efficientdet(phi, num_classes=20, weighted_bifpn=False, freeze_bn=False, score_threshold=0.01):
    assert phi in range(7)
    input_size = image_sizes[phi]
    input_shape = (input_size, input_size, 3)
    # input_shape = (None, None, 3)
    image_input = layers.Input(input_shape)
    w_bifpn = w_bifpns[phi]
    d_bifpn = 2 + phi
    w_head = w_bifpn
    d_head = 3 + int(phi / 3)
    backbone_cls = backbones[phi]
    # features = backbone_cls(include_top=False, input_shape=input_shape, weights=weights)(image_input)
    features = backbone_cls(input_tensor=image_input, freeze_bn=freeze_bn)
    if weighted_bifpn:
        for i in range(d_bifpn):
            features = build_wBiFPN(features, w_bifpn, i, freeze_bn=freeze_bn)
    else:
        for i in range(d_bifpn):
            features = build_BiFPN(features, w_bifpn, i, freeze_bn=freeze_bn)
    regress_head = build_regress_head(w_head, d_head)
    class_head = build_class_head(w_head, d_head, num_classes=num_classes)
    regression = [regress_head(feature) for feature in features]
    regression = layers.Concatenate(axis=1, name='regression')(regression)
    classification = [class_head(feature) for feature in features]
    classification = layers.Concatenate(axis=1, name='classification')(classification)

    model = models.Model(inputs=[image_input], outputs=[regression, classification], name='efficientdet')

    # apply predicted regression to anchors
    # anchors = tf.tile(tf.expand_dims(tf.constant(anchors), axis=0), (tf.shape(regression)[0], 1, 1))
    anchors_input = layers.Input((None, 4))
    boxes = RegressBoxes(name='boxes')([anchors_input, regression])
    boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        name='filtered_detections',
        score_threshold=score_threshold
    )([boxes, classification])
    prediction_model = models.Model(inputs=[image_input, anchors_input], outputs=detections, name='efficientdet_p')
    return model, prediction_model
