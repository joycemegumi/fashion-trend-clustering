read_pxl_id <- function(domain_id, dpt_num_department){
  source("utils/Redshift-connexion_DktCanada.R")
  source("utils/utils.R")
  library(dplyr)
  library(data.table)
  
  # run the query
  dbSendUpdateFromFile(conn, "data/queries/products_pixlID.sql",
                       list_sql_params = list(domain_id=domain_id,
                                              dpt_num_department=dpt_num_department))
  
  # unload pixl IDs and model code to S3
  transactions_s3_path <- paste0("s3://preprod.datamining/images/data/pixlIDs_domain_id_", domain_id, "_dpt_num_department_", dpt_num_department)
  UnloadRedshitToS3(conn, "images", transactions_s3_path, aws_access_key_id, aws_secret_access_key)
  
  dbDisconnect(conn)
}

source("config.R")
domain_id <- '0341'
dpt_num_department <- 371

cc = read_pxl_id(domain_id, dpt_num_department)
