from pymilvus import MilvusClient, DataType, utility, connections



# Establish a connection to the Milvus server
connections.connect(host='localhost', port='19530')
# 1. Set up a Milvus client
client = MilvusClient(
    uri="http://localhost:19530"
)


schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="resource_uri", datatype=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="absolute_url", datatype=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="date_created", datatype=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="date_modified", datatype=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="page_count", datatype=DataType.INT8)
schema.add_field(field_name="download_url", datatype=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="local_path", datatype=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=128)
schema.add_field(field_name="plain_text", datatype=DataType.VARCHAR, max_length=256)
schema.add_field(field_name="html_with_citations", datatype=DataType.VARCHAR, max_length=256)
schema.add_field(field_name="plain_text_vector",
                 datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="html_with_citations_vector",
                 datatype=DataType.FLOAT_VECTOR, dim=768)

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="plain_text_vector", 
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

index_params.add_index(
    field_name="html_with_citations_vector",
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

# 4.3. Create an index file
client.create_index(
    collection_name="llm_law_hackathon_3",
    index_params=index_params
)

client.create_collection(
    collection_name="llm_law_hackathon_3",
    schema=schema,
    index_params=index_params)



tasks = utility.do_bulk_insert(
    collection_name="llm_law_hackathon_2",
    is_row_based=True,
    files=["opinions_flattened_2k_vectorized.json"]
)


state = utility.get_bulk_insert_state(tasks)
print(state.state_name())
print(state.ids())
print(state.infos())