import pybullet as p
import pybullet_data
import math

def main():
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    robotId = p.loadURDF("urdf/aurea_urdf_pkg.urdf", useFixedBase=True)

    print("\n" + "="*70)
    print("URDF (Medidas de Links e Juntas)")
    print("="*70)
    
    for j in range(p.getNumJoints(robotId)):
        info = p.getJointInfo(robotId, j)
        
        id_junta = info[0]
        nome_junta = info[1].decode('utf-8')

        pos_relativa = info[14] 
        
        comprimento = math.sqrt(pos_relativa[0]**2 + pos_relativa[1]**2 + pos_relativa[2]**2)

        print(f"ID: {id_junta:02d} | Junta: {nome_junta:<25} | Vetor(X,Y,Z): {pos_relativa[0]:.4f}, {pos_relativa[1]:.4f}, {pos_relativa[2]:.4f} | Comprimento: {comprimento:.4f} m")

    print("="*70 + "\n")
    p.disconnect()

if __name__ == "__main__":
    main()