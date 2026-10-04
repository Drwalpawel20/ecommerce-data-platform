-- =========================================
-- DATA QUALITY CHECKS
-- =========================================

-- 1. Row counts
SELECT 'dim_category' AS table_name, COUNT(*) AS row_count
FROM warehouse.dim_category

UNION ALL

SELECT 'dim_supplier', COUNT(*)
FROM warehouse.dim_supplier

UNION ALL

SELECT 'dim_employee', COUNT(*)
FROM warehouse.dim_employee

UNION ALL

SELECT 'dim_customer', COUNT(*)
FROM warehouse.dim_customer

UNION ALL

SELECT 'dim_product', COUNT(*)
FROM warehouse.dim_product

UNION ALL

SELECT 'dim_date', COUNT(*)
FROM warehouse.dim_date

UNION ALL

SELECT 'fact_sales', COUNT(*)
FROM warehouse.fact_sales;


-- 2. NULL checks in fact table
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE order_id IS NULL) AS null_order_id,
    COUNT(*) FILTER (WHERE order_detail_id IS NULL) AS null_order_detail_id,
    COUNT(*) FILTER (WHERE date_key IS NULL) AS null_date_key,
    COUNT(*) FILTER (WHERE customer_key IS NULL) AS null_customer_key,
    COUNT(*) FILTER (WHERE product_key IS NULL) AS null_product_key,
    COUNT(*) FILTER (WHERE employee_key IS NULL) AS null_employee_key,
    COUNT(*) FILTER (WHERE quantity IS NULL) AS null_quantity,
    COUNT(*) FILTER (WHERE revenue IS NULL) AS null_revenue,
    COUNT(*) FILTER (WHERE cost IS NULL) AS null_cost,
    COUNT(*) FILTER (WHERE profit IS NULL) AS null_profit
FROM warehouse.fact_sales;


-- 3. Invalid quantities
SELECT COUNT(*) AS invalid_quantity_rows
FROM warehouse.fact_sales
WHERE quantity <= 0;


-- 4. Invalid prices
SELECT COUNT(*) AS invalid_price_rows
FROM warehouse.fact_sales
WHERE unit_price < 0;


-- 5. Invalid discounts
SELECT COUNT(*) AS invalid_discount_rows
FROM warehouse.fact_sales
WHERE discount < 0
   OR discount > 1;


-- 6. Invalid revenue
SELECT COUNT(*) AS invalid_revenue_rows
FROM warehouse.fact_sales
WHERE revenue < 0;


-- 7. Profit calculation validation
SELECT COUNT(*) AS incorrect_profit_rows
FROM warehouse.fact_sales
WHERE ABS(
    profit - (revenue - cost)
) > 0.01;


-- 8. Duplicate order details
SELECT
    order_detail_id,
    COUNT(*) AS occurrences
FROM warehouse.fact_sales
GROUP BY order_detail_id
HAVING COUNT(*) > 1;


-- 9. Products without category
SELECT COUNT(*) AS products_without_category
FROM warehouse.dim_product
WHERE category_key IS NULL;


-- 10. Products without supplier
SELECT COUNT(*) AS products_without_supplier
FROM warehouse.dim_product
WHERE supplier_key IS NULL;


-- 11. Fact rows without valid product
SELECT COUNT(*) AS invalid_product_references
FROM warehouse.fact_sales f
LEFT JOIN warehouse.dim_product p
    ON p.product_key = f.product_key
WHERE p.product_key IS NULL;


-- 12. Fact rows without valid customer
SELECT COUNT(*) AS invalid_customer_references
FROM warehouse.fact_sales f
LEFT JOIN warehouse.dim_customer c
    ON c.customer_key = f.customer_key
WHERE c.customer_key IS NULL;


-- 13. Fact rows without valid employee
SELECT COUNT(*) AS invalid_employee_references
FROM warehouse.fact_sales f
LEFT JOIN warehouse.dim_employee e
    ON e.employee_key = f.employee_key
WHERE e.employee_key IS NULL;


-- 14. Fact rows without valid date
SELECT COUNT(*) AS invalid_date_references
FROM warehouse.fact_sales f
LEFT JOIN warehouse.dim_date d
    ON d.date_key = f.date_key
WHERE d.date_key IS NULL;