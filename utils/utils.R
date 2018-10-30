library(infuser)

dbGetQueryFromFile <- function(conn, file.sql, list_sql_params = list()){
  
  sql_query <- infuse(file.sql, list_sql_params)
  
  data <- dbGetQuery(conn, sql_query)
  
  return(data)
}

dbSendUpdateFromFile <- function(conn, file.sql, list_sql_params = list()){
  
  sql_query <- infuse(file.sql, list_sql_params)
  
  dbSendUpdate(conn, sql_query)
  
}

GetColumnNames <- function(conn, my_table){
  
  col_names <- dbGetQuery(conn, paste0(
    "SELECT column_name FROM information_schema.columns
    WHERE table_name = '",my_table,"'
    ORDER BY ordinal_position;"
  )
  )
  return(unique(col_names))
}

UnloadRedshitToS3 <- function(conn, my_table, s3_path,
                              aws_access_key_id,
                              aws_secret_access_key){
  
  col_names <- GetColumnNames(conn, my_table)
  
  col_names <- col_names %>% 
    dplyr::mutate(column_name_bis = paste0("\\'",column_name, "\\'")) %>% 
    dplyr::mutate(column_casting = paste0("CAST(",column_name, " AS VARCHAR) AS ",column_name))
  
  column_names <- paste(col_names$column_name_bis, collapse = ",")
  column_casting <- paste(col_names$column_casting, collapse = ",")
  
  sql_query <- paste0("SELECT ",column_names,
                      " UNION( SELECT ",column_casting,
                      " FROM ",my_table,") ORDER BY 1 DESC;")
  
  unload_query <- paste0("UNLOAD('",sql_query,"')
                          TO '",s3_path,"' credentials
                          'aws_access_key_id=",aws_access_key_id,";aws_secret_access_key=",aws_secret_access_key,"'  
                          delimiter ';'
                          ESCAPE
                          GZIP
                          ADDQUOTES
                          NULL AS ''
                          parallel off
                          ALLOWOVERWRITE ;")
  
  dbSendUpdate(conn, unload_query)
  
  cat(paste0("table ", my_table, " unloaded to: ", s3_path,"000"))
  
}