/*
-- Script to extract all images ID for products on a decathlon website associated to a given department
*/

--extract the list of products on a web site, based on web tracking information
DROP TABLE if exists web;

CREATE TEMP TABLE web
(
product_id_model varchar(2000),
dpt_num_department bigint
)
DISTKEY(product_id_model)
SORTKEY(product_id_model);

INSERT INTO web
SELECT DISTINCT a.product_id_model, b.dpt_num_department
FROM ODS.DMP_TRACKER_FLOW_AUDIENCE a
INNER JOIN 
(
SELECT DISTINCT mdl_num_model_r3, dpt_num_department
FROM CDS.D_SKU
) b
ON a.product_id_model = b.mdl_num_model_r3
WHERE hit_date > CURRENT_DATE - 30 AND domain_id = {{domain_id}} AND dpt_num_department = {{dpt_num_department}};

-- join the table with the pxl id of the images
DROP TABLE if exists images;

CREATE TEMP TABLE images
(
product_id_model varchar(2000),
id bigint
)
DISTKEY(product_id_model)
SORTKEY(product_id_model);

INSERT INTO images
SELECT DISTINCT a.product_id_model, b.id
FROM web a
INNER JOIN ODS.pxl_products_media_model_code b
ON a.product_id_model = b.model_name
INNER JOIN ODS.pxl_products_media_photo_type c
ON b.id = c.id
WHERE c.photo_type = 1;
