
options(java.parameters = "-Xmx20048m")
options(java.parameters = "-Xss2560k")

require(rJava)
require(RJDBC)
drv <- JDBC("org.postgresql.Driver","~/postgresql-8.4-703.jdbc4.jar",identifier.quote="`")
URL <- "jdbc:postgresql://192.168.112.18:5539/dvdbredshift02"

conn <- dbConnect(drv, URL, user, mdp)

require (dplyr)
# conn_dplyr <- src_postgres(
#   dbname = 'dvdbredshift02',
#   host = 'prdredshift02.c8yxpd5oejlf.eu-west-1.redshift.amazonaws.com',
#   port = 5539,
#   user = user,
#   password = mdp
# )
# 
# 
# rm(mdp)
# rm(user)

#jdbc:postgresql://192.168.112.18:5539/dvdbredshift02

