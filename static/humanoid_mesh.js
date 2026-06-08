import * as THREE from 'three';

const LIMB_SEGMENTS = [
    [11, 13, 0x50d2c2, 0.035],
    [13, 15, 0x50d2c2, 0.028],
    [12, 14, 0xff8068, 0.035],
    [14, 16, 0xff8068, 0.028],
    [23, 25, 0x8bd45f, 0.045],
    [25, 27, 0x8bd45f, 0.036],
    [24, 26, 0xf4bd4f, 0.045],
    [26, 28, 0xf4bd4f, 0.036],
];

const UP = new THREE.Vector3(0, 1, 0);
const TEMP_A = new THREE.Vector3();
const TEMP_B = new THREE.Vector3();
const TEMP_DIR = new THREE.Vector3();
const TEMP_QUAT = new THREE.Quaternion();

function midpoint(a, b) {
    return new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
}

function orientSegment(mesh, a, b) {
    TEMP_A.copy(a);
    TEMP_B.copy(b);
    TEMP_DIR.subVectors(TEMP_B, TEMP_A);
    const length = TEMP_DIR.length();
    if (length < 0.001) {
        mesh.visible = false;
        return;
    }
    mesh.position.copy(TEMP_A);
    mesh.scale.set(1, length, 1);
    TEMP_DIR.normalize();
    TEMP_QUAT.setFromUnitVectors(UP, TEMP_DIR);
    mesh.quaternion.copy(TEMP_QUAT);
    mesh.visible = true;
}

function addSkinningAttributes(geometry) {
    const position = geometry.attributes.position;
    const skinIndices = [];
    const skinWeights = [];
    for (let i = 0; i < position.count; i++) {
        const y = position.getY(i);
        let bone = 0;
        if (y > 0.18) bone = 1;
        if (y > 0.48) bone = 2;
        if (y > 0.72) bone = 3;
        skinIndices.push(bone, 0, 0, 0);
        skinWeights.push(1, 0, 0, 0);
    }
    geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(skinIndices, 4));
    geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(skinWeights, 4));
}

export function createHumanoidMesh() {
    const group = new THREE.Group();
    group.name = 'low_poly_humanoid';

    const hips = new THREE.Bone();
    hips.name = 'hips';
    hips.position.set(0, -0.02, 0);
    const spine = new THREE.Bone();
    spine.name = 'spine';
    spine.position.set(0, 0.28, 0);
    const chest = new THREE.Bone();
    chest.name = 'chest';
    chest.position.set(0, 0.28, 0);
    const head = new THREE.Bone();
    head.name = 'head';
    head.position.set(0, 0.3, 0);
    hips.add(spine);
    spine.add(chest);
    chest.add(head);

    const bodyGeometry = new THREE.CylinderGeometry(0.2, 0.16, 0.9, 8, 8);
    bodyGeometry.translate(0, 0.36, 0);
    addSkinningAttributes(bodyGeometry);

    const bodyMaterial = new THREE.MeshStandardMaterial({
        color: 0xded8c7,
        roughness: 0.62,
        metalness: 0.02,
        transparent: true,
        opacity: 0.72,
        side: THREE.DoubleSide,
    });
    const body = new THREE.SkinnedMesh(bodyGeometry, bodyMaterial);
    const skeleton = new THREE.Skeleton([hips, spine, chest, head]);
    body.add(hips);
    body.bind(skeleton);
    body.castShadow = true;
    group.add(body);

    const headMesh = new THREE.Mesh(
        new THREE.DodecahedronGeometry(0.105, 0),
        new THREE.MeshStandardMaterial({ color: 0xf0d0aa, roughness: 0.55 })
    );
    headMesh.castShadow = true;
    group.add(headMesh);

    const limbMeshes = LIMB_SEGMENTS.map(([, , color, radius]) => {
        const geometry = new THREE.CylinderGeometry(radius, radius * 0.86, 1, 7);
        geometry.translate(0, 0.5, 0);
        const mesh = new THREE.Mesh(
            geometry,
            new THREE.MeshStandardMaterial({
                color,
                roughness: 0.5,
                transparent: true,
                opacity: 0.58,
            })
        );
        mesh.castShadow = true;
        group.add(mesh);
        return mesh;
    });

    group.userData = {
        body,
        hips,
        spine,
        chest,
        head,
        headMesh,
        limbMeshes,
    };
    return group;
}

export function updateHumanoidMesh(group, points, visible = true) {
    if (!group) return;
    const data = group.userData;
    group.visible = visible && Array.isArray(points) && points.length >= 33;
    if (!group.visible) return;

    const leftHip = points[23];
    const rightHip = points[24];
    const leftShoulder = points[11];
    const rightShoulder = points[12];
    const nose = points[0];
    const hipMid = midpoint(leftHip, rightHip);
    const shoulderMid = midpoint(leftShoulder, rightShoulder);
    const torso = new THREE.Vector3().subVectors(shoulderMid, hipMid);
    const torsoLength = Math.max(torso.length(), 0.1);

    data.body.position.copy(hipMid);
    data.body.scale.set(1.0, torsoLength / 0.65, 1.0);
    TEMP_QUAT.setFromUnitVectors(UP, torso.clone().normalize());
    data.body.quaternion.copy(TEMP_QUAT);
    data.hips.position.set(0, 0, 0);
    data.spine.position.set(0, 0.22, 0);
    data.chest.position.set(0, 0.28, 0);
    data.head.position.set(0, 0.24, 0);

    data.headMesh.position.copy(nose);
    data.headMesh.visible = true;

    LIMB_SEGMENTS.forEach(([startIdx, endIdx], i) => {
        orientSegment(data.limbMeshes[i], points[startIdx], points[endIdx]);
    });
}
