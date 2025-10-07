# Get sdv style metadata from tableprocessor

from sdv.metadata import Metadata
from tabkit import ColumnMetadata, TableProcessor


def infer_sdtype(col_info: ColumnMetadata) -> dict[str, str]:
    sdict = {}
    if col_info.kind in ["categorical", "binary"]:
        if col_info.dtype in ["float", "int"]:
            # TODO: if its something like integer flags, should we use
            # numerical or categorical?
            sdict["sdtype"] = "numerical"
        # otherwise, just string
        else:
            sdict["sdtype"] = "categorical"
    elif col_info == "datetime":
        sdict["sdtype"] = "datetime"
        sdict["datetime_format"] = "%s"
    # otherwise, it's probably continuous
    else:
        sdict["sdtype"] = "numerical"
    return sdict


def get_sdv_metadata(proc: TableProcessor) -> Metadata:
    # NOTE: this is what the default is. We will have to re-think this when
    # doing more complex stuff
    table_name = "table"

    # just in case
    proc = proc.prepare()
    df = proc.get("raw_df")
    metadata = Metadata.detect_from_dataframe(df)
    for col in proc.columns_info:
        # grab here because it might be updated in previous iter.
        metadata_cols = metadata.tables[table_name].columns
        inferred_sdtype = infer_sdtype(col)
        if col.name not in metadata_cols:
            metadata.add_column(
                col.name,
                table_name=table_name,
                **inferred_sdtype,
            )
        if (
            inferred_sdtype["sdtype"] != metadata_cols[col.name]["sdtype"]
            # let the id detection work.
            and not (
                inferred_sdtype["sdtype"] == "categorical"
                and metadata_cols[col.name]["sdtype"] == "id"
            )
        ):
            metadata.update_column(
                col.name,
                table_name=table_name,
                **inferred_sdtype,
            )

    assert len(metadata.tables[table_name].columns) == len(df.columns)
    return metadata
