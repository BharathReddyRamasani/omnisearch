def extract_feature_importance(pipeline, X, num_cols, cat_cols):
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return []

    importances = model.feature_importances_

    feature_names = []
    feature_names.extend(num_cols)

    if cat_cols:
        enc = pipeline.named_steps["prep"] \
            .named_transformers_["cat"] \
            .named_steps["encoder"]
        feature_names.extend(enc.get_feature_names_out(cat_cols))

    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    top_features = []
    for f, _ in pairs:
        base = f.split("_")[0]
        if base not in top_features:
            top_features.append(base)

    return top_features[:6]
