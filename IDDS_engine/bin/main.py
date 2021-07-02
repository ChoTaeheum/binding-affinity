import ExpectBA
import ExpectSM
import ToDir
import sys


if __name__ == "__main__":
    req_id = sys.argv[1]
    expect_ba = ExpectBA.ExpectBA(req_id)                        # to_db
    expect_sm = ExpectSM.ExpectSM(req_id)
    
    # results return
    req_id, ba_result                 = expect_ba.get_result()
    sm_result, file_list, ligand_info = expect_sm.get_result()   # to_db
    
    ToDir.ToDir(req_id, ba_result, sm_result, file_list, ligand_info)