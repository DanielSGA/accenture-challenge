import config from './config.json'
const aws = require('aws-sdk')

(async function() {
    try {
        aws.config.setPromisesDependency();
        aws.config.update({
            accessKeyId: config.aws.accessKey,
            secretAccessKey: config.aws.secretKey,
        });

        const s3 = new aws.S3();
        const response = await s3.listObjectsV2({
            Bucket: 'acn-bucket',
            Prefix: 'plates'
        }).promise();

    } catch (e) {
        console.log('Error', e)
    }
})();

export default Response;