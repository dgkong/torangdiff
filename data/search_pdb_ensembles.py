import json

import requests


def search_pdb_protein_ensembles(query: json, output_file: str) -> None:
    """
    https://search.rcsb.org/#search-api
    """
    search_api_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    print("Executing search query...")
    response = requests.post(search_api_url, json=query)
    
    if response.status_code == 200:
        results = response.json()
        print(f"Successfully found {len(results['group_set'])} total groups.")
        ensembles_found = 0
        with open(output_file, "w") as f_out:
            for group in results["group_set"]:
                if group['count'] > 1:
                    full_ids = list(set([member['identifier'] for member in group['result_set']]))
                    pdb_ids = list(set([entity.split("_")[0] for entity in full_ids]))
                    if len(pdb_ids) > 1: # filter clusters with entries that come from the same PDB entry
                        representatives = [] # pick one representative entity per PDB entry
                        seen_pdb_ids = set()
                        for full_id in sorted(full_ids):
                            pdb_id = full_id.split("_")[0]
                            if pdb_id not in seen_pdb_ids:
                                representatives.append(full_id)
                                seen_pdb_ids.add(pdb_id)
                        f_out.write(",".join(representatives) + "\n")
                        ensembles_found += 1
        print(f"Found and saved {ensembles_found} ensembles to '{output_file}'.")
    else:
        print(f"Failed to search PDB. Status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    query_file = "data/search_pdb_query.json"
    ensembles_file = "data/pdb_ensembles.txt"
    with open(query_file, "r") as f:
        query = json.load(f)
    search_pdb_protein_ensembles(query, ensembles_file)
