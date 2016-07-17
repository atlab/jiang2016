import datajoint as dj

schema = dj.schema('share_jiang2016_connectivity', locals())

schema.spawn_missing_classes()