-- Schema para el producto de datos de demanda en retail (1C Company)
-- Ejecutar una vez contra la instancia RDS provisioned por rds.yaml
-- Idempotente: usa IF NOT EXISTS en cada CREATE TABLE

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. products — catálogo maestro de productos por tienda
--    Un registro por par (item_id, shop_id).
--    Tabla raíz: todas las demás tienen FK hacia aquí.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS products (
    id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id    VARCHAR(20)  NOT NULL,
    category   VARCHAR(100),
    shop_id    VARCHAR(20)  NOT NULL,
    active     BOOLEAN      NOT NULL DEFAULT TRUE,
    UNIQUE (item_id, shop_id)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. predictions — salida del modelo para Nov-2015 (date_block_num=34).
--    predicted_sales, lower_bound y upper_bound en unidades mensuales (0-20).
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id              UUID      PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id      UUID      NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    prediction_date DATE      NOT NULL,
    predicted_sales FLOAT     NOT NULL,
    lower_bound     FLOAT,
    upper_bound     FLOAT,
    generated_at    TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. actuals — ventas históricas reales (Oct-2015, mes 33).
--    Una fila por par (product_id, sale_date).
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS actuals (
    id            UUID  PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id    UUID  NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    sale_date     DATE  NOT NULL,
    actual_sales  FLOAT NOT NULL
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. evaluation_metrics — métricas de evaluación del modelo por grupo.
--    product_id es NULL para métricas agregadas (global, category:X).
--    group_key identifica el nivel: 'global' | 'category:N' | 'item:N,shop:M'
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id  UUID         REFERENCES products(id) ON DELETE CASCADE,
    group_key   VARCHAR(200) NOT NULL,
    rmse        FLOAT        NOT NULL,
    mae         FLOAT        NOT NULL,
    naive_rmse  FLOAT        NOT NULL
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. business_feedback — retroalimentación del negocio sobre predicciones.
--    sentiment: 'positivo' | 'negativo' | 'neutro'
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS business_feedback (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id  UUID         NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    sentiment   VARCHAR(20)  NOT NULL CHECK (sentiment IN ('positivo', 'negativo', 'neutro')),
    observation TEXT,
    created_by  VARCHAR(100),
    created_at  TIMESTAMP    NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 6. flagged_products — productos marcados para revisión.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS flagged_products (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id  UUID         NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    feedback_id UUID         REFERENCES business_feedback(id) ON DELETE SET NULL,
    reason      TEXT         NOT NULL,
    resolved    BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMP    NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- ÍNDICES
-- ─────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_products_item       ON products(item_id);
CREATE INDEX IF NOT EXISTS idx_products_shop       ON products(shop_id);
CREATE INDEX IF NOT EXISTS idx_products_category   ON products(category);
CREATE INDEX IF NOT EXISTS idx_predictions_product ON predictions(product_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date    ON predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_actuals_product     ON actuals(product_id);
CREATE INDEX IF NOT EXISTS idx_actuals_date        ON actuals(sale_date);
CREATE INDEX IF NOT EXISTS idx_eval_product        ON evaluation_metrics(product_id);
CREATE INDEX IF NOT EXISTS idx_eval_group          ON evaluation_metrics(group_key);
CREATE INDEX IF NOT EXISTS idx_feedback_product    ON business_feedback(product_id);
CREATE INDEX IF NOT EXISTS idx_feedback_sentiment  ON business_feedback(sentiment);
CREATE INDEX IF NOT EXISTS idx_flagged_product     ON flagged_products(product_id);
CREATE INDEX IF NOT EXISTS idx_flagged_resolved    ON flagged_products(resolved);
