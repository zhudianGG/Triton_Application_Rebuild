import argparse

def replace_replica_pb(config_path, replica):
    from google.protobuf import text_format
    with open(config_path, "r") as fr:
        text_format.Parse(fr.read(), obj)
    obj["instance_group"][0]["count"] = replica
    with open(config_path, "w") as fw:
        text_format.PrintMessage(obj, fw)

def replace_replica(config_path, replica):
    with open(config_path, "r") as fr:
        obj_str = fr.read()
    start_pos = obj_str.find("instance_group")
    if start_pos < 0:
        print(f"error not find instance_group: {obj_str}")
        return
    find_str = "count:"
    start_pos = obj_str.find(find_str, start_pos)
    if start_pos < 0:
        print(f"error not find instance_group2: {obj_str}")
        return
    end_pos = obj_str.find("\n", start_pos)
    if end_pos < 0:
        print(f"error not find instance_group end_pos: {obj_str}")
        return
    obj_str = obj_str[:start_pos] + find_str + f" {replica}" + obj_str[end_pos:]
    with open(config_path, "w") as fw:
        fw.write(obj_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rewrite_config_pbtext')
    parser.add_argument('--replica', default = 1)
    parser.add_argument('--path', required = True)
    #parser.add_argument('--port',default = 10030)
    args = parser.parse_args()
    replace_replica(args.path, args.replica)